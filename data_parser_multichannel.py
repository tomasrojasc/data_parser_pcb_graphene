import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def get_header_dict_raw(text_file):
    """This function takes the text file and extracts a list with the header

    Args:
        text_file (str): the string of the whole file

    Returns:
        tuple: tuple with the dict of parameters of the header and the last index of the header
    """
    pattern_header = re.compile(r"[+-]?((\d+(\.\d+)?)|(\.\d+))\n")
    header_matches = pattern_header.finditer(text_file)
    header = [float(i.group()[:-1]) for i in header_matches]
    header_dic = dict(timestamp=header[0],
                  setted_v_d=header[1],
                  temp_1=header[2],
                  temp_2=header[3],
                  temp_3=header[4],
                  read_v_d=header[5],
                  read_v_plus=header[6],
                  setted_v_g=header[7],
                  setted_div_steps=header[8],
                  delta_t=header[9],
                  channels=int(header[10]),
                  samples_per_point=header[11]
                  )
    pattern_header = re.compile(r"[+-]?((\d+(\.\d+)?)|(\.\d+))\n")
    header_matches = pattern_header.finditer(text_file)
    end_of_header = [i.span() for i in header_matches]
    end_of_header = end_of_header[-1][-1]
    
    return header_dic, end_of_header


def get_data_raw(text_file, end_of_header):
    """This function takes the raw text file string and the index on which the header ends and returns a dataframe
    with the unprocessed data

    Args:
        text_file (str): string which is the raw text file
        end_of_header (int): integer representing where the header ends

    Returns:
        DataFrame: pandas dataframe with unprocessed data
    """
    csvfile = StringIO(text_file[end_of_header:])  # this line saves the string as a csv locally
    df_raw = pd.read_csv(csvfile, header=None)
    df_raw = df_raw.iloc[: , :-1]  # dropping last column since is empty
    return df_raw


def get_channels(decimal_repr):
    """This function takes a decimal representation of the channels and turns it into a list with all used channels

    Args:
        decimal_repr (int): decimal representation of the combination of channels

    Returns:
        list: list with the decimal representation of each channel used
    """
    bin_repr = [int(i) for i in list(bin(decimal_repr))[2:]]
    len_list = len(bin_repr)
    channels = [len_list - i - 1 for i, j in enumerate(bin_repr) if j==1]
    return channels
    
    # [0, 1, 1, 1] valores
    # [0, 1, 2, 3] indices
    # [3, 2, 1, 0] canales
    # largo = 4
    # canal = largo - indice -1


def get_data_names(channels):
    """This function generates the names of the current values, both the currents and the std and returns a list containing all

    Args:
        channels (list): list with the channels to

    Returns:
        list: list with the names of the columns
    """
    channel_mapper = {"2": "AorB", "3": "CorD", "4": "EorF", "6":"GorH", "7":"IorJ"}
    names_values = ["channel_" + channel_mapper[str(i)] for i in channels][::-1]
    names_std = ["channel_" + channel_mapper[str(i)] + "_std" for i in channels][::-1]
    names = [val for pair in zip(names_values, names_std) for val in pair]
    
    return names

def get_names_based_on_version(df, channels):
    n_cols = len(df.columns)
    
    try:
        assert n_cols in [12, 13, 16]
    except AssertionError:
        print("The file version could not be identified")
    
    if n_cols == 12:  # we have a v1 file
        names = ["setted_v_g", "read_v_g"] + get_data_names(channels)
    elif n_cols == 13:  # we have a v2 file
        names = ["datetime", "setted_v_g", "read_v_g"] + get_data_names(channels)
    elif n_cols == 16:  # we have a v3 file
        names = ["datetime", "temp1", "temp2", "temp3", "setted_v_g", "read_v_g"] + get_data_names(channels)
    
    return names


def rename_raw_df(df, channels):
    """This function renames all the columns in a dataframe

    Args:
        df (DataFrame): dataframe containing the data of the experiment
        channels (list): decimal representation of the used channels

    Returns:
        DataFrame: dataframe with the renamed columns
    """
    
    names = get_names_based_on_version(df, channels)
    mapper = {i: names[i] for i in range(len(names))}
    df = df.rename(mapper=mapper, axis=1)
    return df


def get_raw_header_and_data(text_file):
    """This function takes the text file of the data and returns two dataframes: one for the metadata
    or header and one for the actual data.

    Args:
        text_file (str): string with the data read from a file

    Returns:
        tuple: tuple with two dataframes, one for the header and one for the data
    """
    header_dic, end_of_header = get_header_dict_raw(text_file)
    data_raw = get_data_raw(text_file, end_of_header)
    
    channels = get_channels(header_dic["channels"])
    
    data_raw = rename_raw_df(data_raw, channels)
    raw_header_df = pd.DataFrame(header_dic, index=["values"]).T 
    
    return raw_header_df, data_raw


def convert_current_ds(curr):
    """This function converts the counts unit of the measurements to actual currents

    Args:
        curr (int): count representation (digital) of the current

    Returns:
        float: actual current
    """
    return curr * 31.2666171 + 391.129525


def convert_volt_g(volt):
    """This function converts the counts unit of the measurements to actual gate voltages

    Args:
        curr (int): count representation (digital) of the gate voltages

    Returns:
        float: actual gate voltage
    """
    return (volt - 2042.8342) / 81.8877461


def convert_volt_d(volt):
    """This function converts the counts unit of the measurements to actual drain voltages

    Args:
        curr (int): count representation (digital) of the drain voltages

    Returns:
        float: actual drain voltage
    """
    return volt * 0.0006103 + 0.0045770


def convert_volt_plus(volt):
    """This function converts the counts unit of the measurements to actual plus voltages

    Args:
        curr (int): count representation (digital) of the plus voltages

    Returns:
        float: actual plus voltage
    """
    # need edit
    return volt * 10 * 5.0 / (2 ** 12)
 
    
def convert_metadata(raw_metadata_df):
    """This function takes the raw metadata and applies all the conversions

    Args:
        raw_metadata_df (DataFrame): pandas dataframe with the raw metadata

    Returns:
        DataFrame: pandas dataframe with the converted data
    """
    
    raw_metadata_df_T = raw_metadata_df.T

    # individual variables    
    timestamp = raw_metadata_df_T["timestamp"]
    setted_v_d = raw_metadata_df_T["setted_v_d"]
    temp_1 = raw_metadata_df_T["temp_1"]
    temp_2 = raw_metadata_df_T["temp_2"]
    temp_3 = raw_metadata_df_T["temp_3"]
    read_v_d = raw_metadata_df_T["read_v_d"]
    read_v_plus = raw_metadata_df_T["read_v_plus"]
    setted_v_g_max = raw_metadata_df_T["setted_v_g"]
    setted_div_steps = raw_metadata_df_T["setted_div_steps"]
    delta_t = raw_metadata_df_T["delta_t"]
    channels = int(raw_metadata_df_T["channels"])
    samples_per_point = raw_metadata_df_T["samples_per_point"]
    
    # convertions
    c_timestamp = pd.to_datetime(timestamp ,unit='s')
    c_setted_v_d = setted_v_d 
    c_temp_1 =temp_1
    c_temp_2=temp_2
    c_temp_3=temp_3
    c_read_v_d = convert_volt_d(read_v_d)
    c_read_v_plus = convert_volt_plus(read_v_plus)
    c_setted_v_g_max = setted_v_g_max
    c_setted_div_steps = int(setted_div_steps)
    c_delta_t = delta_t
    c_channels = get_channels(channels)
    c_samples_per_point = int(samples_per_point)
    
    # converted dict
    
    converted_dict = dict(timestamp=c_timestamp,
                          setted_v_d=c_setted_v_d,
                          temp_1=c_temp_1,
                          temp_2=c_temp_2,
                          temp_3=c_temp_3,
                          read_v_d=c_read_v_d,
                          read_v_plus=c_read_v_plus,
                          setted_v_g=c_setted_v_g_max,
                          setted_div_steps=c_setted_div_steps,
                          delta_t=c_delta_t,
                          channels=str(c_channels),
                          samples_per_point=c_samples_per_point,
                          )
    
    return pd.DataFrame(converted_dict, index=["values"]).T
    

def f(row, col_names=None):
    """This function is intended to use with pandas mapper to transform a dataframe's data in an efficient way

    Args:
        row (?): row at which we are now in the dataframe
        col_names (list, optional): list with the names of the columns. Defaults to None.

    Returns:
        ?: new row with the convertions applied
    """

    if len(row) == 12:  # version 1 file
        volt_names = col_names[:2]  # columns which are voltage related
        curr_names = col_names[2:]  # columns which are current related
        
        volt_row = row[volt_names]  # take only the values for the voltages
        curr_row = row[curr_names]  # take only the values for the currents
        
        converted_volt = convert_volt_g(volt_row)  # converting voltages
        converted_curr = convert_current_ds(curr_row)  # converting currents
        
        a = pd.concat([converted_volt, converted_curr], sort=False, join="inner")  # joining everythong together
        
    elif len(row) == 13:  # version 2 file
        
        time_names = col_names[0]  # columns with timestamp
        volt_names = col_names[1:3]  # columns which are voltage related
        curr_names = col_names[3:]  # columns which are current related
        
        volt_row = row[volt_names]  # take only the values for the voltages
        curr_row = row[curr_names]  # take only the values for the currents
        time_row = row[time_names]  # take only the values of timestamps
        
        converted_volt = convert_volt_g(volt_row)  # converting voltages
        converted_curr = convert_current_ds(curr_row)  # converting currents
        converted_time = pd.Series(pd.to_datetime(time_row, unit="s"))  # converting timestamp
        
        a = pd.concat([converted_time, converted_volt, converted_curr], sort=False, join="inner")  # joining everythong together

    elif len(row) == 16:  # version 3 file

        time_names = col_names[0]  # columns with timestamp
        temp_names = col_names[1:4]  # columns with temperatures
        volt_names = col_names[4:6]  # columns which are voltage related
        curr_names = col_names[6:]  # columns which are current related
        
        time_row = row[time_names]  # take only the values of timestamps
        temp_row = row[temp_names]  # take only the values of temperatures
        volt_row = row[volt_names]  # take only the values for the voltages
        curr_row = row[curr_names]  # take only the values for the currents   
        

        converted_time = pd.Series(pd.to_datetime(time_row, unit="s"))  # converting timestamp
        converted_volt = convert_volt_g(volt_row)  # converting voltages
        converted_curr = convert_current_ds(curr_row)  # converting currents
        converted_temp = temp_row
        
        a = pd.concat([converted_time, temp_row, converted_volt, converted_curr], sort=False, join="inner")  # joining everythong together

    
    return a

def get_version(n_cols):
    if n_cols == 12:
        return "v1"
    elif n_cols == 13:
        return "v2"
    elif n_cols == 16:
        return "v3"

def convert_data(raw_data_df):
    """This functions applies the f mapper to a dataframe

    Args:
        raw_data_df (DataFrame): raw dataframe

    Returns:
        DataFrame: new converted dataframe
    """
    return raw_data_df.apply(f, col_names=list(raw_data_df.columns), axis=1)


class Data:
    def __init__(self, file_name):
        assert type(file_name) == str
        
        self.file_name = file_name
        with open(self.file_name) as file:
            self.raw_data = file.read()
        
        
        self.raw_metadata, self.raw_data = get_raw_header_and_data(self.raw_data)  # here we create the two raw dataframes
        
        # waiting convertions
        self.converted_data_df = None
        self.converted_metadata = None
        self.temp_df = None
        
        # converting the data
        self.convert_data()
        self.convert_metadata()
        
        self.n_cols = len(self.converted_data_df.columns)
        self.fotmat_version = get_version(self.n_cols)
        
        if self.n_cols == 16:
            self.temp_df = self.converted_data_df.iloc[:, :4]
        
    def convert_data(self):
        self.converted_data_df = convert_data(self.raw_data)
        return
    
    def convert_metadata(self):
        self.converted_metadata = convert_metadata(self.raw_metadata)
    
        
        
    def plot_converted(self, filename=None, figsize=None, xlim=None, initial_marker=True, grid=None, *args, **kwargs):
        if isinstance(figsize, type(None)):
            plt.figure(*args, **kwargs)
        else:
            plt.figure(figsize=figsize,*args, **kwargs)
        plt.title("$I_{ds}$ as a function of $V_g$")
        plt.xlabel("$V_g$ [V]")
        plt.ylabel("$I_{ds}$ [$\mu$A]")
        
        if self.n_cols == 12:  # version 1 file
            x = self.converted_data_df.iloc[:,1].values
            y = self.converted_data_df.iloc[:,2::2].values
            cols = list(self.converted_data_df.iloc[:,2::2].columns)
            
        elif self.n_cols == 13:  # version 2 file
            x = self.converted_data_df.iloc[:,2].values
            y = self.converted_data_df.iloc[:,3::2].values
            cols = list(self.converted_data_df.iloc[:,3::2].columns)
            
        elif self.n_cols == 16:  # version 3 file
            x = self.converted_data_df.iloc[:,5].values
            y = self.converted_data_df.iloc[:,6::2].values
            cols = list(self.converted_data_df.iloc[:,6::2].columns)

        plt.plot(x, y / 1e3)
        plt.legend(cols)
        
        
        
        if not isinstance(xlim, type(None)):
            plt.xlim(xlim)
        
        if grid:
            plt.grid()


       # if initial_marker:
         #   x, y = self.converted_data_df.read_adc_volt_g.values, self.converted_data_df.curr_value_ds.values / 1000.
          #  plt.plot(x[0], y[0],'or')

        if not isinstance(filename, type(None)):
            plt.savefig(filename + ".png", dpi=300)
        return
    
    def plot_temp(self):
        if self.fotmat_version != "v3":
            print("This version of data does not support this method :(, you need a v3 file for this")
            print("current version: {self.format_version}")
            return
            
        self.temp_df.plot(x=0)
        return
    