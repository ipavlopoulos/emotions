import pandas as pd
import os


class DataHandler:
    """
    this class serves to store data from the Streamer and give data to the visualizer
    """
    def __init__(self, directory, max_capacity_per_file):
        """

        :param directory: the directory where all csv paths will be stored
        :param max_capacity_per_file: how many rows a file can have. If a file reach this limit, then a new file
        stores the rest of the data.
        """
        self.id_prefix = "_id_"
        self.directory = directory
        self.max_capacity_per_file = max_capacity_per_file
        self.create_dir_if_not_exists()
        self.capacity_on_last_path = self.get_capacity_of_latest_path()
        self.latest_path_id = self.get_latest_path_id()

    def get_latest_path_id(self) -> int:
        """
        find the id of the last csv file. The next file will have the next natural number as id
        :return:
        """
        paths = [x for x in os.listdir(self.directory) if x.endswith("csv")]
        if not len(paths):
            latest = 0
        else:
            ids = [int(x.split(self.id_prefix)[-1].split(".")[0]) for x in paths if x.endswith(".csv")]
            latest = max(ids)
            if self.max_capacity_per_file - len(self.get_dataframe_by_id(latest)) == 0:
                latest += 1
        return latest

    def get_capacity_of_latest_path(self) -> int:
        """
        How may rows can store the latest path.
        :return: the difference between the maximum capacity in rows of the last path and the current size
        """
        path = self.get_latest_path()
        if not os.path.exists(path):
            return self.max_capacity_per_file
        else:
            return self.max_capacity_per_file - len(pd.read_csv(path, lineterminator='\n'))

    def get_latest_path(self) -> str:
        """
        :return: the latest path
        """
        return self.get_path_by_id(self.get_latest_path_id())

    def get_path_by_id(self, num):
        return os.path.join(self.directory, "tweets" +
                            self.id_prefix +
                            str(num) +
                            ".csv")

    def get_dataframe_by_id(self, num):
        return pd.read_csv(self.get_path_by_id(num), lineterminator='\n') if os.path.exists(self.get_path_by_id(num)) else pd.DataFrame()

    def get_latest_dataframe(self):
        return self.get_dataframe_by_id(self.get_latest_path_id())

    def create_dir_if_not_exists(self) -> None:
        """
        is called at the initialization of the object in order to check if exists the directory where the csv files
        will be saves exists
        :return: None
        """
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def dump_on_new_file(self, new_data: pd.DataFrame) -> None:
        """
        create new csv and save data
        :param new_data: dict {str:list}
        :return:
        """
        new_data.to_csv(self.get_latest_path(), index=False)
        self.check_and_update_capacity()

    def append(self, df_old: pd.DataFrame, new_data: pd.DataFrame):
        """
        :param df_old: a pd.Dataframe with stored data
        :param new_data:
        :return:
        """
        df = pd.concat([df_old, new_data])
        df.to_csv(self.get_latest_path(), index=False)
        self.check_and_update_capacity()

    def check_and_update_capacity(self):
        """
        When a csv reaches the full capacity, then prepares for the opening of the next file
        :return:
        """
        capacity = self.get_capacity_of_latest_path()
        if capacity == 0:
            self.latest_path_id += 1
            self.capacity_on_last_path = self.max_capacity_per_file

    def store_new_data(self, df_new: pd.DataFrame):
        """
        runs the whole operation of storing new data
        :param df_new: the new data as pandas Dataframe
        :return:
        """
        capacity = self.get_capacity_of_latest_path()
        if len(df_new) < capacity:
            if not os.path.exists(self.get_latest_path()):
                self.dump_on_new_file(df_new)
            else:
                df_old = self.get_latest_dataframe()
                self.append(df_old, df_new)
        else:
            split1 = df_new.iloc[0: capacity]
            split2 = df_new.iloc[capacity:]
            df_old = self.get_latest_dataframe()
            self.append(df_old, split1)
            self.dump_on_new_file(split2)

    def load_all_data(self) -> pd.DataFrame:
        """
        load all data
        """
        df = self.get_dataframe_by_id(0)
        latest = self.get_latest_path_id()
        for idx in range(1, latest+1):
            df = pd.concat([df, self.get_dataframe_by_id(idx)])
        return df
