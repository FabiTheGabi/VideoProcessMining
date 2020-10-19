# Created in the VideoProcessMining project
import csv
import datetime
import os
import numpy as np
import pandas as pd

from pm4py.objects.log.adapters.pandas import csv_import_adapter
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.util import constants

import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
from slowfast.utils.ava_eval_helper import read_labelmap

logger = logging.get_logger(__name__)


class DemoMeter(object):
    """
    Logs the detections from the demo
    """

    def __init__(self, cfg, img_height, img_width):
        """
        Initialize the DemoMeter with the relevant paramters
        :param cfg:
        :param img_height: (int) the height of the input images
        :param img_width: (int) the width of input images
        """

        # Set up environment.
        setup_environment()

        self.cfg = cfg

        # In the case of an AVA-like predictor it is necessary to specify a
        # cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE, because it comprises all 80 categories.
        # during the ava challenge only 60 categories were evaluated, an we
        # want alle categories
        path_to_label_map_file = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.LABEL_MAP_FILE) \
            if not os.path.isfile(cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE) \
            else cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE

        # Export properties
        datetime_for_filenames = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.delimiter = ","
        assert cfg.DEMO.OUTPUT_FOLDER != "", "Please specify cfg.DEMO.OUTPUT_FOLDER to be able to export the output"
        self.output_dir_path = self.cfg.DEMO.OUTPUT_FOLDER
        self.file_name_demo_log = datetime_for_filenames + "_" + "demo_log"
        self.file_name_demo_gt_like_file = datetime_for_filenames + "_" + "demo_gt_format"
        self.results_gt_like_csv_path = os.path.join(self.output_dir_path, self.file_name_demo_gt_like_file + ".csv")
        self.results_log_path_prefix = os.path.join(self.output_dir_path, self.file_name_demo_log)
        self.results_log_csv_path = ""
        self.results_xes_path = ""
        # The minimum score for a predicted category to be exported
        self.min_category_export_score = cfg.DEMO.EXPORT_MIN_CATEGORY_EXPORT_SCORE
        # Whether a person can do multiple actions at the same time or not
        # This influences the export options, since only the option with max value is chosen
        self.multiple_action_possible = cfg.CUSTOM_DATASET.MULTIPLE_ACTION_POSSIBLE

        # Resolution used for export
        self.img_height = img_height
        self.img_width = img_width

        # List of dict with items "id" and "name"
        self.categories, _ = read_labelmap(path_to_label_map_file)
        # Replace delimiter out of category_name to guarantee good csv export
        for idx in range(0, len(self.categories)):
            self.categories[idx]["name"] = self.categories[idx]["name"].replace(self.delimiter, "")

        # The list-variables we use to store the demo prediction results.
        # They will be used to export the information into csv or xes
        self.res_person_tracking_outputs = []
        self.res_pred_action_category_scores = []
        self.res_all_metadata = []
        self.res_case_ids = []

        # This df is used to assign correct case_concept_name and concept_instance values
        self.case_and_instance_df = self.create_empty_case_and_instance_df()
        # Used to indicate that the activity instance has not yet completed
        self.video_second_not_complete = -1

        # Used as test for correct functionality of lifecycle_transition
        self.not_closed_list = []

    def create_empty_case_and_instance_df(self):
        """
        Creates the empty df
        :return:
        """
        return pd.DataFrame(columns=("video_id", "video_second_start", "video_second_complete",
                                                          "org_resource", "category_name", "case_concept_name",
                                                          "concept_instance_key"))

    def add_detection(self, video_id, video_second, person_tracking_outputs, pred_action_category_scores, case_ids=-1):
        """
        Saves all relevant information about a detection for a specific video and second
        :param video_id: (int) the id of the video, starting from 0
        :param video_second: (int) the second of the video, for which predictions were computed
        :param person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
                                  --> if empty it is a list []
        :param pred_action_category_scores: pred_action_category_scores (ndarray float32)
                                                shape(num_person_ids, num_categories),
                                               the scores for each person and each action category
                                           --> if empty it is a list []
       :param case_ids: int64 ndarray with shape (num_identities, 1 = case_id): contains the case_ids for an action
                            of a person
        """

        self.res_person_tracking_outputs.append(person_tracking_outputs)
        self.res_pred_action_category_scores.append(pred_action_category_scores)

        # We will check the case_ids. We will create an empty list [] or an ndarray shape (num_identities, 1 = case_id)
        # If no pred_action_category_scores are provided, case_ids are an empty list.
        # Otherwise, they will be assigned the default value 0

        # No case_ids provided as argument
        if isinstance(case_ids, int):
            if isinstance(pred_action_category_scores, list):
                # case_ids has also to be an empty list
                case_ids = []
            else:
                # Fill case_ids with the default value 0
                case_ids = []
                for person in range(0, pred_action_category_scores.shape[0]):
                    case_ids.append(0)
                case_ids = np.asanyarray(case_ids, dtype=np.int64)
        elif isinstance(case_ids, list):
            assert len(case_ids) == len(
                pred_action_category_scores), "Please make sure that case_ids is also [], when pred_action_category_scores is []"
        else:
            assert case_ids.shape[0] == pred_action_category_scores.shape[
                0], "You have to provide a case id for every person"

        self.res_case_ids.append(case_ids)

        # Add the metadata entry
        metadata_entry = {"video_id": video_id, "video_second": video_second}
        self.res_all_metadata.append(metadata_entry)

        assert len(self.res_person_tracking_outputs) == len(self.res_pred_action_category_scores)
        assert len(self.res_person_tracking_outputs) == len(self.res_all_metadata)
        assert len(self.res_case_ids) == len(self.res_all_metadata)

    def export_results(self):
        """
        Exports the data to csv file
        :return:
        """

        # Export the unmodified results
        self.export_results_to_ground_truth_like_file()

        # Modify the self.res_pred_action_category_scores, so that only the highest score remains
        if not self.multiple_action_possible:
            self.modify_res_pred_action_category_scores_for_single_action()

        while self.min_category_export_score <= 1:
            self.export_results_to_process_log_csv()
            # We have to export to csv first, because we build or xes export based on the csv
            self.export_results_to_process_log_xes()
            self.min_category_export_score = self.min_category_export_score + 0.05

    def export_results_to_ground_truth_like_file(self):
        """
        This function exports the results to a csv file that is similar to the files specified in
        CUSTOM_DATASET.TRAIN_GT_BOX_LISTS or CUSTOM_DATASET.GROUNDTRUTH_FILE
        The csv filefile has the following columns:
        1) Video_ID (e.g.. vid1), 2) Middle_frame_timestamp (in seconds from video start), 3) x1, 4) y1, 5) x2,
        6) y2 (3-6 nomalized with respect to frame size), 7) category_id: (e.g. 1), 8) category_name,
        9) category_score, 10) Person_id
        :return:
        """

        # write the new csv file
        with open(self.results_gt_like_csv_path, "w") as demo_results_csv:
            writer = csv.writer(demo_results_csv, quotechar="'", delimiter=self.delimiter)
            writer.writerow(
                ["video_id", "Middle_frame_timestamp", "x1", "y1", "x2", "y2", "category_id", "category_name",
                 "category_score", "person_id", "case_id"])

            # We have stored all our predictions for every second with predicted values
            for second_idx in range(0, len(self.res_person_tracking_outputs)):
                res_person_tracking_outputs = self.res_person_tracking_outputs[second_idx]
                res_pred_action_category_scores = self.res_pred_action_category_scores[second_idx]
                metadata = self.res_all_metadata[second_idx]
                case_ids = self.res_case_ids[second_idx]

                video_id = metadata.get("video_id")
                Middle_frame_timestamp = metadata.get("video_second")

                for person_tracking_outputs, res_pred_action_category_scores, case_id in zip(
                        res_person_tracking_outputs,
                        res_pred_action_category_scores, case_ids):
                    # The timestamp for the video
                    x1 = '{:.3f}'.format(round(person_tracking_outputs.data[0] / self.img_width, 3))
                    y1 = '{:.3f}'.format(round(person_tracking_outputs.data[1] / self.img_height, 3))
                    x2 = '{:.3f}'.format(round(person_tracking_outputs.data[2] / self.img_width, 3))
                    y2 = '{:.3f}'.format(round(person_tracking_outputs.data[3] / self.img_height, 3))
                    person_id = person_tracking_outputs.data[4]

                    for category_score, category_idx in zip(res_pred_action_category_scores,
                                                            range(0, res_pred_action_category_scores.shape[0])):
                        # Get correct category attributes
                        category_id = self.categories[category_idx]["id"]
                        category_name = self.categories[category_idx]["name"]
                        # Extract prediction score and round to 5 digits
                        category_score = '{:.5f}'.format(category_score, 5)
                        # Add line to csv file
                        writer.writerow(
                            [video_id, Middle_frame_timestamp, x1, y1, x2, y2, category_id, category_name,
                             category_score, person_id, case_id])

        logger.info("Exported groundtruth like_file to: %s" % self.results_gt_like_csv_path)

    def modify_res_pred_action_category_scores_for_single_action(self):
        """
        Modifies self.res_pred_action_category_scores, so that only the score(s) with the highest value
        remain. This is done for every second and person. All scores not having the maximum value are assigned a
        negative value to ensure that they are not exported
        :return:
        """
        # We have stored all our predictions for every second with predicted values
        for second_idx in range(0, len(self.res_pred_action_category_scores)):
            # Get all detections for the second
            res_pred_action_category_scores_per_second = self.res_pred_action_category_scores[second_idx]

            person_idx = 0
            # Get all persons for the second
            for res_pred_action_category_scores_per_second_an_person in res_pred_action_category_scores_per_second:
                category_idx = 0
                for single_action_category_score in res_pred_action_category_scores_per_second_an_person:
                    if single_action_category_score < max(res_pred_action_category_scores_per_second_an_person):
                        self.res_pred_action_category_scores[second_idx][person_idx, category_idx] = -0.1
                    category_idx = category_idx + 1
                person_idx = person_idx + 1

    def export_results_to_process_log_csv(self):
        """
        Exports the prediction results of the demo to a process log-like csv file
        It is possible to filter for category predictions above
        a threshold >= self.min_category_export_score
        We also export process mining specific attributes
        :return:
        """

        # Create filename for current self.min_category_export_score
        self.results_log_csv_path = self.results_log_path_prefix + "_" + \
                                         str(int(self.min_category_export_score * 100)) + "_thresh.csv"
        # Create empty df
        self.case_and_instance_df = self.create_empty_case_and_instance_df()

        # The required attributes for process mining
        # Contains the case ID, in our 0 is the default
        case_concept_name = 0
        # The unique id of an event
        event_id = 1

        # The artificial timestamp when assuming the
        # start of the video was today at 00:00
        time_timestamp = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # write the new csv file
        with open(self.results_log_csv_path, "w") as demo_results_csv:
            writer = csv.writer(demo_results_csv, quotechar="'", delimiter=self.delimiter)
            writer.writerow(
                ["video_id", "video_second", "x1", "y1", "x2", "y2", "category_score", "category_id", "category_name",
                 "case:concept:name", "concept:name", "time:timestamp", "org:resource", "eventid",
                 "lifecycle:transition", "concept:instance"])

            # We have stored all our predictions for every second with predicted values
            for second_idx in range(0, len(self.res_person_tracking_outputs)):
                res_person_tracking_outputs = self.res_person_tracking_outputs[second_idx]
                res_pred_action_category_scores = self.res_pred_action_category_scores[second_idx]
                metadata = self.res_all_metadata[second_idx]
                case_ids = self.res_case_ids[second_idx]

                video_id = metadata.get("video_id")
                video_second = metadata.get("video_second")

                for person_tracking_outputs, res_pred_action_category_scores, case_id in zip(
                        res_person_tracking_outputs,
                        res_pred_action_category_scores, case_ids):
                    # The timestamp for the video
                    time_timestamp_for_video_second = time_timestamp + datetime.timedelta(seconds=video_second)
                    x1 = person_tracking_outputs.data[0]
                    y1 = person_tracking_outputs.data[1]
                    x2 = person_tracking_outputs.data[2]
                    y2 = person_tracking_outputs.data[3]
                    org_resource = person_tracking_outputs.data[4]
                    case_concept_name = case_id

                    for category_score, category_idx in zip(res_pred_action_category_scores,
                                                            range(0, res_pred_action_category_scores.shape[0])):
                        if category_score >= self.min_category_export_score:
                            # Get lifecycle_info and whether we will output this event
                            lifecycle_transition, include_event_in_export = self.get_lifecycle_transition_info(
                                second_idx,
                                org_resource,
                                category_idx)
                            # We only write start and complete events no "intermediate" events
                            if include_event_in_export:
                                # ToDo: delete if everything works
                                not_closed_entry = str(org_resource) + "|" + str(category_idx)
                                if lifecycle_transition == "start":
                                    self.not_closed_list.append(not_closed_entry)
                                else:
                                    if not_closed_entry in self.not_closed_list:
                                        self.not_closed_list.remove(not_closed_entry)

                                # Get correct category attributes
                                category_id = self.categories[category_idx]["id"]
                                category_name = self.categories[category_idx]["name"]
                                # Extract prediction score and round to 5 digits
                                category_score = '{:.5f}'.format(category_score, 5)

                                # Adjust concept name and concept instance according tho this logic
                                case_concept_name_list, concept_instance_key_list, lifecycle_transition_info_list = self.get_case_id_and_concept_instance(
                                    video_id=video_id,
                                    video_second=video_second,
                                    org_resource=org_resource,
                                    category_name=category_name,
                                    lifecycle_transition_info=lifecycle_transition,
                                    current_case_concept_name=case_concept_name)

                                for case_concept_name, concept_instance, lifecycle_transition in zip(case_concept_name_list, concept_instance_key_list, lifecycle_transition_info_list):
                                    # Add line to csv file
                                    writer.writerow(
                                        [video_id, video_second, x1, y1, x2, y2, category_score, category_id, category_name,
                                         case_concept_name, category_name, time_timestamp_for_video_second, org_resource,
                                         event_id, lifecycle_transition, concept_instance])
                                    # Increase the event_id
                                    event_id += 1
        # ToDo: delete if everything works
        assert len(self.not_closed_list) == 0, "Logic fault in start and complete logic"

        logger.info("Exported demo detections to: %s" % self.results_log_csv_path)

    def get_case_id_and_concept_instance(self, video_id, video_second, org_resource, category_name,
                                         lifecycle_transition_info, current_case_concept_name):
        """
        This function handels the assignment of case_id and concept_instance.
        The logic behind it is, that two events belonging to the same activity_instance have the same
        case_id (an activity instance belongs to only one case) and concept_instance (to distinguish them from
        other similar activites that may occur in the meantime).

        The concept_instance key is derived as follows:
        video_id + "|" + str(video_second) + "|" + str(org_resource) + "|" + category_name

        Furthermore, we assure that every activity instance has a start and complete lifecycle
        transition. This is only relevant if it starts and ends in the same second.

        :param video_id: (str) the name of the video
        :param video_second: (int) the video second to which this event belongs to
        :param org_resource: (int) the assigned id of the org_resource
        :param category_name: (str) the name of the event
        :param lifecycle_transition_info: (string) "", "complete", or "start"
        :param current_case_concept_name: the name of the case the event belongs to
        :return:
            Three lists with len num events 1 (start or complete or 2 (start and complete)
            case_concept_name: list (str): the case:concept:name for the input data
            concept_instance_key: list (str): the concept:instance for the input data
            lifecycle_transition_info: list (str): "start" or "complete"
        """
        case_concept_name_list = []
        concept_instance_key_list = []
        lifecycle_transition_info_list = []
        create_start_and_end_event = False

        # These are the return values, except a start event already exists
        case_concept_name = current_case_concept_name
        concept_instance_key = video_id + "|" + str(video_second) + "|" + str(org_resource) + "|" + category_name

        assert lifecycle_transition_info in ["start", "complete"], "Warning lifecycle_transition_info is not allowed"
        if lifecycle_transition_info == "start":
            # Add entry to df
            self.add_entry_to_case_and_instance_df(video_id, video_second, org_resource,
                                                   category_name, case_concept_name,
                                                   concept_instance_key)
        elif lifecycle_transition_info == "complete":

            key_exits, case_concept_name_from_start_event, concept_instance_key_from_start_event = \
                self.get_case_concept_name_and_instance_key_from_start_event(video_id, org_resource, category_name)

            if key_exits:
                assert (case_concept_name_from_start_event != "" and concept_instance_key_from_start_event != ""), \
                    "Error in Logic: these values must not be empty"
                case_concept_name = case_concept_name_from_start_event
                concept_instance_key = concept_instance_key_from_start_event
            else:
                # add this concept instance key -> we have an activity instance that start and ends in the same second
                # ->we have to create two events with lifecycle transition start and complete and the same second
                self.add_entry_to_case_and_instance_df(video_id, video_second, org_resource,
                                                       category_name, case_concept_name,
                                                       concept_instance_key)
                create_start_and_end_event = True

            # in any case set entry complete
            self.set_complete_case_and_instance_df(concept_instance_key, video_second)

        if create_start_and_end_event:
            # The start event
            case_concept_name_list.append(case_concept_name)
            concept_instance_key_list.append(concept_instance_key)
            lifecycle_transition_info_list.append("start")

            # The complete event
            case_concept_name_list.append(case_concept_name)
            concept_instance_key_list.append(concept_instance_key)
            lifecycle_transition_info_list.append("complete")
        else:
            # Use original data only
            case_concept_name_list.append(case_concept_name)
            concept_instance_key_list.append(concept_instance_key)
            lifecycle_transition_info_list.append(lifecycle_transition_info)

        return case_concept_name_list, concept_instance_key_list, lifecycle_transition_info_list

    def add_entry_to_case_and_instance_df(self, video_id, video_second_start, org_resource,
                                          category_name, case_concept_name, concept_instance_key):
        """
        Adds a new entry to the df
        :param video_id: (str) the name of the video
        :param video_second_start: (int) the video second to which this event belongs to (always the start second)
        :param org_resource: (int) the assigned id of the org_resource
        :param category_name: (str) the name of the event
        :param case_concept_name: the name of the case the event belongs to
        :param concept_instance_key: (str) the key for the concept instance
        :return:
        """
        key_list = self.case_and_instance_df["concept_instance_key"].tolist()
        key_in_key_list = concept_instance_key in key_list

        assert not key_in_key_list, "Concept instance key already exists"

        # When adding a row, video_second_complete is always self.video_second_not_complete
        # Prepare the entry
        new_entry = [video_id, video_second_start, self.video_second_not_complete, org_resource, category_name,
                     case_concept_name, concept_instance_key]
        # Add the new entry
        self.case_and_instance_df.loc[len(self.case_and_instance_df)] = new_entry

    def set_complete_case_and_instance_df(self, concept_instance_key, video_second_complete):
        """
        Marks that an activity instance has completed by setting video_second_complete
        :param concept_instance_key: (str) the key for the concept instance
        :param video_second_complete: (int) the second when the activity instance completes
        :return:
        """
        self.case_and_instance_df.loc[(self.case_and_instance_df["concept_instance_key"] == concept_instance_key),
                                      "video_second_complete"] = video_second_complete

    def get_case_concept_name_and_instance_key_from_start_event(self, video_id, org_resource, category_name):
        """
        Searches for a start event (with self.video_second_not_complete and returns the corresponding information
        :param video_id: (str) the name of the video
        :param org_resource: (int) the assigned id of the org_resource
        :param category_name: (str) the name of the event
        :return:
            start_event_and_key_exits: (boolean) True if a key exists, False if not an the key has to be created
            case_concept_name: (str) the concept name of the case
            concept_instance_key: (str) the respective key for the input parameters
        """
        key_exits = False
        case_concept_name = ""
        concept_instance_key = ""

        # Filter the entries based on the attributes. We want start events belonging to activity instances
        # that have not completed yet
        filtered_df = self.case_and_instance_df.loc[(self.case_and_instance_df["video_id"] == video_id) &
                                                    (self.case_and_instance_df["org_resource"] == org_resource) &
                                                    (self.case_and_instance_df["category_name"] == category_name) &
                                                    (self.case_and_instance_df[
                                                         "video_second_complete"] == self.video_second_not_complete)]

        len_filtered_df = len(filtered_df.index)
        assert len_filtered_df in [0, 1], "There is an error in self.case_and_instance_df"

        # If a start event exists, we get the correct values
        if len_filtered_df == 1:
            case_concept_name = filtered_df["case_concept_name"].values[0]
            concept_instance_key = filtered_df["concept_instance_key"].values[0]
            key_exits = True

        return key_exits, case_concept_name, concept_instance_key

    def get_lifecycle_transition_info(self, second_idx, org_resource, category_idx):
        """
        Determines whether an event has lifecycletransition complete or start
        If it is none of both, we wil not include the event in the export
        :param second_idx: (int) the index of the current video second in our result data
        :param org_resource: (int) the id of the resource who conducted the event
        :param category_idx: (int) the index of the category that determines the event that we export
        :return:
            org_resource: (string) "", "complete", or "start"
            include_event_in_export: (boolean) whether we will export this event (only if start or complete)
        """

        lifecycle_transition = ""
        include_event_in_export = False

        # If an event is a complete and start event at the same time, we will always choose the lifecycle_transition
        # complete
        if self.is_complete_lifecycle_transition(second_idx, org_resource, category_idx):
            lifecycle_transition = "complete"
            include_event_in_export = True
        elif self.is_start_lifecycle_transition(second_idx, org_resource, category_idx):
            lifecycle_transition = "start"
            include_event_in_export = True

        return lifecycle_transition, include_event_in_export

    def is_complete_lifecycle_transition(self, second_idx, org_resource, category_idx):
        """
        Determines if an event has lifecycle_transition complete. This is the case if the org_resource
        has no activity with category_idx or no activity with category_idx and >= self.min_category_export_score for
        the next_second_idx
        :param second_idx: (int) the index of the current video second in our result data
        :param org_resource: (int) the id of the resource who conducted the event
        :param category_idx: (int) the index of the category that determines the event that we export
        :return:
            True: if the event has lifecycle_transition complete
            False: else
        """
        next_second_idx = second_idx + 1

        # If this is the last index, the lifecycle_transition is complete
        if second_idx == (len(self.res_person_tracking_outputs) - 1):
            return True
        # If we do not have any person detections for the next observation, the lifecycle_transition is complete
        elif len(self.res_person_tracking_outputs[next_second_idx]) == 0:
            return True
        # Otherwise we have to check the next_second_idx in detail
        else:
            # Get the org resources
            org_resources_next_second = self.res_person_tracking_outputs[next_second_idx][:, 4]

            # If the org resource is in the next second, we have to check the score for the predicted activities
            if org_resource in org_resources_next_second:
                corresponding_row_index_next_second = np.where(org_resources_next_second == org_resource)
                # From ndarray to int
                corresponding_row_index_next_second = corresponding_row_index_next_second[0][0]
                # Get the score_tresh the org resource has for the next_second_idx for category_idx
                category_score_next_second = self.res_pred_action_category_scores[next_second_idx][
                    corresponding_row_index_next_second, category_idx]

                # we will not export the next second/category_idx combination --> the lifecycle_transition is complete
                if not category_score_next_second >= self.min_category_export_score:
                    return True
                # we will export the next second/category_idx combination --> the current second is not complete
                else:
                    return False
            # We do not have any data for this org resource for the next second --> the lifecycle_transition is complete
            else:
                return True

    def is_start_lifecycle_transition(self, second_idx, org_resource, category_idx):
        """
        Determines if an event has lifecycle_transition sart. This is the case if the org_resource
        has no activity with category_idx or no activity with category_idx and >= self.min_category_export_score for
        the previous_second_idx
        :param second_idx: (int) the index of the current video second in our result data
        :param org_resource: (int) the id of the resource who conducted the event
        :param category_idx: (int) the index of the category that determines the event that we export
        :return:
            True: if the event has lifecycle_transition start
            False: else
        """
        previous_second_idx = second_idx - 1

        # If this is the first index, the lifecycle_transition is start
        if second_idx == 0:
            return True
        # If we do not have any person detections for the previous observation, the lifecycle_transition is start
        elif len(self.res_person_tracking_outputs[previous_second_idx]) == 0:
            return True
        # Otherwise we have to check the previous_second_idx
        else:
            # Get the org resources
            org_resources_previous_second = self.res_person_tracking_outputs[previous_second_idx][:, 4]

            # If the org resource is in the previous second, we have to check in detail
            if org_resource in org_resources_previous_second:
                corresponding_row_index_previous_second = np.where(org_resources_previous_second == org_resource)
                # From ndarray to int
                corresponding_row_index_previous_second = corresponding_row_index_previous_second[0][0]
                # Get the score_tresh the org resource has for the next_second_idx for category_idx
                category_score_previous_second = self.res_pred_action_category_scores[previous_second_idx][
                    corresponding_row_index_previous_second, category_idx]

                # we have not exported the previous second/category_idx combination
                # --> the lifecycle_transition is start
                if not category_score_previous_second >= self.min_category_export_score:
                    return True
                # we have exported any previous second/category_idx combination as start
                # --> the current second is not start
                else:
                    return False
            # We do not have any data for this org resource for the previous second
            # --> the lifecycle_transition is start
            else:
                return True

    def export_results_to_process_log_xes(self):
        """
        Exports the results of the demo to an xes file
        with the same name as the previously exported csv file.
        We load this csv file and transform it into an xes file.
        :return:
        """

        # Create filename for current self.min_category_export_score
        self.results_xes_path = self.results_log_path_prefix + "_" + \
                                         str(int(self.min_category_export_score * 100)) + "_thresh.xes"


        # Read previously generated csv file an transform to log
        dataframe = csv_import_adapter.import_dataframe_from_path(
            self.results_log_csv_path, sep=",")
        log = conversion_factory.apply(dataframe,
                                       parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",
                                                   constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "category_name",
                                                   constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})

        # Add relevant data for ProM import
        log._classifiers = {'Event Name': ['concept:name'],'(Event Name AND Lifecycle transition)': ['concept:name', 'lifecycle:transition']}
        log._extensions = {'Time': {'prefix': 'time', 'uri': 'http://www.xes-standard.org/time.xesext'}, 'Lifecycle': {'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'}, 'Concept': {'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}}

        for trace in log._list:
            # set trace concept:name to str instead of int, also for ProM import
            trace._attributes["concept:name"] = str(trace._attributes["concept:name"])
            # Set org:resource to string as well
            for item in trace._list:
                item["org:resource"] = str(item["org:resource"])

        # Export results to xes
        xes_exporter.export_log(log, self.results_xes_path)

        logger.info("Exported demo detections to: %s" % self.results_xes_path)