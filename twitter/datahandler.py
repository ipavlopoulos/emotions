import pandas as pd
import os
from datetime import datetime


class DataHandler:
    """
    this class serves to store data from the Streamer and give data to the visualizer
    """
    def __init__(self, directory):
        """

        :param directory: the directory where all csv paths will be stored
        """
        self.directory = directory

    def store_new_data(self, df_new: pd.DataFrame):
        """
        runs the whole operation of storing new data
        :param df_new: the new data as pandas Dataframe
        :return:
        """
        current_date = str(datetime.date(datetime.now()))
        path = os.path.join(self.directory, current_date+".csv")
        if not os.path.exists(path):
            df_new.to_csv(path, index=False)
        else:
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new])
            df.to_csv(path, index=False)

    def load_all_data(self) -> pd.DataFrame:
        """
        load all data
        """
        df = pd.concat([pd.read_csv(os.path.join(self.directory, x)) for x in os.listdir(self.directory)])
        return df
