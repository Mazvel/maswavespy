# -*- coding: utf-8 -*-
#
#    MASWavesPy, a Python package for processing and inverting MASW data
#    Copyright (C) 2023  Elin Asta Olafsdottir (elinasta(at)hi.is)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MASWavesPy wavefield/dispersion/combination

Create a Dataset object to initiate and manage multiple ElementDC objects 
for a given seismic dataset (where each ElementDC object represents a single
multi-channel record) and initialize a CombineDCs object from the set of 
identified elementary dispersion curves. 

"""

import pandas as pd
import pickle

from maswavespy.wavefield import RecordMC
from maswavespy.combination import CombineDCs


class Dataset():

    """
    Class for creating dataset objects.
    
    Instance attributes
    -------------------
    site : str
        Name of test site.
    profile : str 
        Identification code for measurement profile.
    date : str
        Date(s) of measurement.
    fs : float
        Sampling frequency [Hz].
    f_pick_min : int or float
        Lower boundary of the geophones' responce curve [Hz].
    metadata : str or dict
        Meta-information for the dataset.
    records : dict
        Dictionary of multi-channel record (RecordMC) objects.
        The keys of records are the record IDs of the imported shot gathers.
    element_dcs : dict
        Dictionary of elementary dispersion curve (ElementDC) objects.
        The keys of element_dcs are the record IDs of the imported shot gathers.
    dcs : combination.CombineDCs
        Composite dispersion curve (CombineDCs) object.
    
    
    Instance methods
    ----------------
    add_from_textfile(self, record_id, file_name, header_lines, n, direction, dx, x1, **kwargs)
        Add a multi-channel record from a text file to a dataset object. 
        
    add_from_waveform(self, record_id, file_name, n, direction, dx, x1, **kwargs)
        Add a multi-channel record from a waveform file to a dataset object. 
    
    add_record(self, record_id, traces, n, direction, dx, x1, file_name=None)
        Add a multi-channel record to a dataset object.     
    
    delete_record(self, record_id)
        Delete a particular multi-channel record (ElementDC object) from
        a dataset object.
    
    get_dcs(self, record_ids='all', metadata=None)
        Create a dictionary of identified elementary dispersion curves and
        initialize a composite dispersion curve object.
    
    records_from_csv(self, import_format, csv_file)
        Import a set of multi-channel records from a comma-separated values (CSV) file.  
    
    save_to_pickle(self, saveas_filename)
        Pickle a dataset object.
    
    _check_record_id(self, record_id)
        Ensure that each imported shot gather holds a unique record ID.
        
    _list_of_keys(self, record_ids)
        Retrieve or check list of record IDs.

    
    Static methods
    --------------
    _check_format(import_format)
        Ensure that the import format of data files is specified as 'textfile' 
        or 'waveform'.
        
    _open_csv_file(import_format, csv_file)
        Open CSV file as a dataframe and ensure that the imported data fulfils 
        all requirements for further analysis.
        
    """
           
    def __init__(self, site, profile, date, fs, f_pick_min, metadata=None):
        
        """
        Initialize a dataset object.
        
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
        date : str    
            Date(s) of measurement.
        fs : float
            Sampling frequency [Hz].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve (depends on 
            the geophones' natural frequency) [Hz]. Spectral maxima 
            (experimental dispersion curves) are only identified at 
            frequencies higher than f_pick_min.
        metadata : str or dict, optional
            Meta-information for object. Additional information about the
            survey configuration/recorded data.
            Default is metadata=None.
            
        Returns
        -------
        Dataset
            Initialized dataset object.
            
        """
        self.site = site
        self.profile = profile
        self.date = date
        self.fs = fs
        self.f_pick_min = f_pick_min
        self.metadata = metadata
        
        self.records = {}
        self.element_dcs = {}
        self.dcs = None
        
        
    def _check_record_id(self, record_id):
        
        """ 
        Ensure that each imported shot gather holds a unique record 
        identification (record ID).
        
        Parameters
        ----------
        record_id : str, int or float
            Record identification (ID).
        
        Returns
        -------
        record_id : str, int or float
            Checked record identification (ID).
        
        Raises
        ------
        ValueError
            If a record with ID record_id is already present in the dataset.
        
        """
        if record_id in self.records.keys():
            message = f'Each record must hold a unique ID. A record with ´record_id´ {record_id} is already present in the dataset.'
            raise ValueError(message)
        else:
            return record_id
        

    def add_record(self, record_id, traces, n, direction, dx, x1, file_name=None):
        
        """
        Add a multi-channel surface wave record to a dataset object. 
        
        The record is added as a multi-channel record object
        to the dictionary (instance attribute) records.
        
        Parameters
        ----------
        record_id : str, int or float
            Record identification (ID). A unique string or number used as 
            a record ID for each imported shot gather.
        traces : numpy.ndarray
            Multi-channel surface wave record.
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'} 
            Direction of measurement.            
            - 'forward': Forward measurement.
              Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse' : Reverse (backward) measurement.
              Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        file_name : str, optional
            Path of file with recorded seismic data. 
            Default is file_name=None.
            
        """
        record_id = self._check_record_id(record_id)

        # Add as a multi-channel record object to ´records´ 
        self.records[record_id] = RecordMC(self.site, self.profile, traces, n, direction, dx, x1, 
                                            self.fs, self.f_pick_min, file_name=file_name)
        
        
    def add_from_textfile(self, record_id, file_name, header_lines, n, direction, dx, x1, **kwargs):
        
        """
        Add a multi-channel surface wave record (stored in a text file) to  
        a dataset object. 
        
        The record is added as a multi-channel record object to the dictionary 
        (instance attribute) records.
        
        Parameters
        ----------
        record_id : str, int or float
            Record identification (ID). A unique string or number used as 
            a record ID for each imported shot gather.
        file_name : str
            Path of file with recorded seismic data.
            Requirements:    
            - By default, it is assumed that the recorded data is stored in 
              a whitespace/tab-delimited text file. If a different string is 
              used to separate values, it can be passed to the loadtxt command 
              as an additional keyword argument (e.g., delimiter=',' for 
              a comma-separated textfile).
            - The number of header lines, including comments, is header_lines.
            - Each trace must be stored in a single column.
            - All traces must be of equal length and missing values are not 
              allowed (each row in the text file must include the same number
              of values).
        header_lines : int
            Number of header lines, including comments. 
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'}  
            Direction of measurement.
            - 'forward': Forward measurement.
              Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse' : Reverse (backward) measurement.
              Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
            
        Other parameters
        ----------------
        Other keyword arguments are passed on to numpy.loadtxt.
        See https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
        for a list of valid kwargs for the loadtxt command. 

        """
        record_id = self._check_record_id(record_id)
        
        # Add as a multi-channel record object to ´records´     
        self.records[record_id] = RecordMC.import_from_textfile(self.site, self.profile, file_name, header_lines, n, direction, 
                      dx, x1, self.fs, self.f_pick_min, **kwargs)
        
        
    def add_from_waveform(self, record_id, file_name, n, direction, dx, x1, **kwargs):
       
        """
        Add a multi-channel surface wave record (stored in a waveform file) 
        to a dataset object. 
        
        The record is added as a multi-channel record object to the dictionary 
        (instance attribute) records.
        
        Parameters
        ----------
        record_id : str, int or float
            Record identification (ID). A unique string or number used as 
            a record ID for each imported shot gather.
        file_name : str
            Path of file with recorded seismic data.  
            See https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
            for a list of supported file formats.
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'}  
            Direction of measurement.
            - 'forward': Forward measurement.
              Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse' : Reverse (backward) measurement.
              Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].    
            
        Other parameters
        ----------------
        Other keyword arguments are passed on to obspy.core.stream.read.
        See https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
        for a list of valid kwargs for the obspy.read command.
        
        """
        record_id = self._check_record_id(record_id)
        
        # Add as a multi-channel record object to ´records´ 
        self.records[record_id] = RecordMC.import_from_waveform(self.site, self.profile, file_name, n, direction, 
                      dx, x1, self.fs, self.f_pick_min, **kwargs)

                
    def delete_record(self, record_id):
        
        """
        Delete a particular multi-channel surface wave record (RecordMC 
        object) from a dataset object.

        Parameters
        ----------
        record_id : str, int or float
            Record identification (ID). A unique string or number used as 
            a record ID for each imported shot gather.

        Raises
        ------
        KeyError
            If a record with ID record_id is not included in the dataset.

        """
        if record_id in self.records:
            del self.records[record_id]
        else:
            message = f'A record with ´record_id´ {record_id} was not found in the dataset.'
            raise KeyError(message)
      
            
    @staticmethod
    def _check_format(import_format):
        
        """
        Ensure that the import format of data files is specified as 'textfile' 
        or 'waveform'.

        Parameters
        ----------
        import_format : str
            Import format of data files, 'textfile' or 'waveform'.

        Returns
        -------
        import_format : {'textfile', 'waveform'}
            Checked import format. If the string 'import_format' contains 
            upper case letters they are converted to lower case.  

        Raises
        ------
        ValueError
            If import_format is not specified as 'textfile' or 'waveform'.

        """
        formats = ['textfile', 'waveform']
        if import_format.lower() not in formats:
            message = f'import_format must be specified as ´textfile´ or ´waveform´, not as ´{import_format}´'
            raise ValueError(message)
        else:
            return import_format.lower()
    
    
    @staticmethod
    def _open_csv_file(import_format, csv_file):
        
        """
        Open CSV file and ensure that its contents fulfil the following
        criteria:
            
        - import_type is 'textfile' 
           Required headings are 'record_id', 'file_name', 'header_lines', 
           'n', 'direction', 'dx' and 'x1'. The data types (pandas dtype) 
           for each of the columns must be as follows:
               - 'record_id' : object, int64 or float64
               - 'file_name' : object
               - 'header_lines' : int64
               - 'n' : int64
               - 'direction' : object
               - 'dx' : int64 or float 64
               - 'x1' : int64 or float 64
               
        - import_type is 'waveform' 
           Required headings are 'record_id', 'file_name', 'n', 'direction', 
           'dx' and 'x1'. The data types (pandas dtype) for each of the 
           columns must be as follows:
               - 'record_id' : object, int64 or float64
               - 'file_name' : object
               - 'n' : int64
               - 'direction' : object
               - 'dx' : int64 or float 64
               - 'x1' : int64 or float 64

        Parameters
        ----------
        import_format : {'textfile', 'waveform'}
            Import format.
        csv_file : str
            Path of CSV file.

        Returns
        -------
        df : Pandas dataframe
            Imported file.
            
        Raises
        ------
        ValueError
            If one or more columns of csv_file do not have the correct heading(s) 
            or if required columns are missing.
        TypeError
            If incorrect data types are found in csv_file.
        
        """
        df = pd.read_csv(csv_file)

        # Check headings
        header_list = list(df)
        if import_format == 'textfile':  
            required_headings = ['record_id', 'file_name', 'header_lines', 'n', 'direction', 'dx', 'x1']
        else:
            required_headings = ['record_id', 'file_name', 'n', 'direction', 'dx', 'x1']
        
        in_required_headings = [elem in header_list for elem in required_headings]
        if not all(in_required_headings):
            ids = [elem for elem, value in enumerate(in_required_headings) if not value]
            if len(ids) == 1:
                message = f'A column with heading {required_headings[ids[0]]} was not found in the imported CSV file.'  
            else:
                missing_headings = ', '.join([str(required_headings[i]) for i in ids])
                message = f'Columns with headings {missing_headings} were not found in the imported CSV file.'
            raise ValueError(message)
            
        # Check data types (pandas dtype)
        data_types = df.dtypes
        required_data_type = []
        required_data_type.append(data_types['record_id'] in ['object', 'int64', 'float64'])
        required_data_type.append(data_types['file_name'] == 'object')
        required_data_type.append(data_types['n'] == 'int64')
        required_data_type.append(data_types['direction'] == 'object')
        required_data_type.append(data_types['dx'] in ['int64', 'float64'])
        required_data_type.append(data_types['x1'] in ['int64', 'float64'])
        if import_format == 'textfile':  
            required_data_type.append(data_types['header_lines'] == 'int64')
        if not all(required_data_type):
            message = 'Incorrect data types found in the imported CSV file.'
            raise TypeError(message)
        else:
            return df
       
        
    def records_from_csv(self, import_format, csv_file):
        
        """
        Batch import a set of multi-channel surface wave records from 
        a comma-separated values (CSV) file.       

        Parameters
        ----------
        import_format : {'textfile', 'waveform'}
            Import format.
            
            - 'textfile'
               Multi-channel surface wave records are added from text files.
               For further documentation, see Dataset.add_from_textfile and
               dispersion.ElementDC.from_textfile.
            - 'waveform'
               Multi-channel surface wave records are added from waveform files.
               For further documentation, see Dataset.add_from_waveform and
               dispersion.ElementDC.from_waveform.
               
        csv_file : str
            Path of CSV file.
                
            - import_type is 'textfile' 
               Required headings are 'record_id', 'file_name', 'header_lines', 
               'n', 'direction', 'dx' and 'x1'. The data types (pandas dtype) 
               for each of the columns must be as follows:
                   - 'record_id' : object, int64 or float64
                   - 'file_name' : object
                   - 'header_lines' : int64
                   - 'n' : int64
                   - 'direction' : object
                   - 'dx' : int64 or float 64
                   - 'x1' : int64 or float 64
                   
            - import_type is 'waveform' 
               Required headings are 'record_id', 'file_name', 'n', 'direction', 
               'dx' and 'x1'. The data types (pandas dtype) for each of the 
               columns must be as follows:
                   - 'record_id' : object, int64 or float64
                   - 'file_name' : object
                   - 'n' : int64
                   - 'direction' : object
                   - 'dx' : int64 or float 64
                   - 'x1' : int64 or float 64

        """
        import_format = self._check_format(import_format)
        df = self._open_csv_file(import_format, csv_file) 
        no_records = len(df) 
        
        if import_format == 'textfile':
            for i in range(no_records):
                self.add_from_textfile(df['record_id'][i], df['file_name'][i], 
                                       df['header_lines'][i], df['n'][i],  
                                       df['direction'][i], df['dx'][i], df['x1'][i])
        else:
            for i in range(no_records):
                self.add_from_waveform(df['record_id'][i], df['file_name'][i], df['n'][i],  
                                       df['direction'][i], df['dx'][i], df['x1'][i])
    
    
    def _list_of_keys(self, record_ids):
        
        """
        Retrieve or check list of record IDs.
        
        - If record_ids is 'all', return a list of record IDs in the dataset  
          (i.e., a list of the keys in the element_dcs dict).
        - If record_ids is a list, ensure that all elements in the list have  
          been defined as record IDs for the dataset (each element in 
          the list must also be a key in the element_dcs dict).
        
        Parameters
        ----------
        record_ids : str or list
            List of record IDs or 'all' (to retrieve all record IDs).

        Returns
        -------
        list
            Checked list of record IDs.
            
        Raises
        ------
        ValueError
            If an element in record_ids does not exist as a key in the element_dcs dict. 
        
        """
        if isinstance(record_ids, str) and record_ids.lower() == 'all':
            return list(self.element_dcs.keys())
        else:
            for key in record_ids:
                if key not in list(self.element_dcs.keys()):
                    message = f'An ElementDC object with ´record_id´ {key} not found in the dataset.'
                    raise ValueError(message)
            return record_ids
            
    
    def get_dcs(self, record_ids='all', to_CombineDCs=True, metadata=None):
        
        """
        Create a dictionary of identified elementary dispersion curves. 
        Each curve is expressed as a dictionary containing the elementary 
        dispersion curve and general information on the measurement site 
        and profile configuration. See dispersion.return_to_dict for further information.
            
        Initialize a composite dispersion curve (CombineDCs) object from the dictionary of 
        elementary dispersion curves, optional. See combination.import_from_dict for  
        further information.

        Parameters
        ----------
        record_ids : str or list, optional
            List of record IDs or 'all' (to retrieve all record IDs).
            Default is record_ids='all'.
        to_CombineDCs : boolean, optional
            Initializes a composite dispersion curve object if set as True.
            Default is to_CombineDCs = True
        metadata : str or dict, optional
            Meta-information for object. Default is metadata=None.
        
        Return
        ------
        element_dc_dict : dict
            Dictionary of identified elementary dispersion curves. 
        
        """
        # Retrieve/check list of record IDs
        record_ids = self._list_of_keys(record_ids)
        
        # Get elementary dispersion curves
        element_dc_dict = {}
        for record_id in record_ids:
            element_dc_dict[record_id] = self.element_dcs[record_id].return_to_dict()
        
        # Initialize a composite dispersion curve object
        if to_CombineDCs:
            self.dcs = CombineDCs.import_from_dict(element_dc_dict, metadata=metadata)
    
        return element_dc_dict


    def save_to_pickle(self, saveas_filename):
        
        """
        Pickle a dataset object.
        
        Parameters
        ----------
        saveas_filename : str
            Write the pickled representation of the dataset object to the 
            file object file.
        
        """
        if ( saveas_filename[-2:] != '.p' and saveas_filename[-7:] != '.pickle' ):
            saveas_filename += '.p'
        
        pickle_out = open(saveas_filename, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()    
        
    