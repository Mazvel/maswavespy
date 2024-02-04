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
MASWavesPy Dispersion

Identify (elementary) experimental dispersion curves from multi-channel
surface wave registrations. The data processing is conducted by 
application of the phase-shift method (Park et al. 1998). The dispersion 
analysis computational tools are based on those presented in 
Olafsdottir et al. (2018).

References
----------
Phase-shift method
 - Park, C.B., Miller, R.D. and Xia J. (1998). Imaging dispersion curves 
   of surface waves on multi-channel record. In SEG technical program 
   expanded abstracts 1998, New Orleans, LA, pp. 1377–1380. 
   https://doi.org/10.1190/1.1820161
MASWaves (MATLAB implementation)
 - Olafsdottir, E.A., Erlingsson, S. and Bessason, B. (2018). 
   Tool for analysis of multichannel analysis of surface waves (MASW) 
   field data and evaluation of shear wave velocity profiles of soils. 
   Canadian Geotechnical Journal 55(2): 217–233. 
   https://doi.org/10.1139/cgj-2016-0302
   
"""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class SelectDC(tk.Frame):
    
    """
    A GUI application to identify (elementary) dispersion curves from
    imported multi-channel surface wave registrations.
    
    Inheritance
    -----------
    Inherents from tkinter.Frame.
    
    
    Instance attributes
    -------------------
    ax0 : axes object (matplotlib axes)
        The axes of the dispersion image (subplot).

    ax0.paths_clicks : list
        Dispersion curve identification using mouse click events. 
        PathCollection objects for clicked points.

    ax0.paths_ids : list
        Dispersion curve identification using window selection and/or spectral 
        maxima IDs. PathCollection object for selected spectral maxima.

    ax0.paths_labels : list
        List of spectral maxima labels (text instances).

    ax0.paths_maxima : list
        PathCollection object for spectral maxima.

    canvas : FigureCanvas (matplotlib canvas)
        The canvas the dispersion image renders into. 

    cid_mouse : int
        Callback id. Dispersion curve identification by clicking on 
        the dispersion image, connect to mouse button press events.

    cid_window_quit : int 
        Callback id. Window selection of spectral maxima, connect to  
        mouse button release events.

    cid_window_start : int
        Callback id. Window selection of spectral maxima, connect to  
        mouse button press events.

    cid_window_update: int
        Callback id. Window selection of spectral maxima, connect to 
        mouse motion notify events.

    dc_clicks : tuple
        Coordinates of dispersion curve data points (obtained by clicking on 
        the dispersion image/mouse click events). 
        dc_clicks[0] is a list of frequency values [Hz]. 
        dc_clicks[1] is a list of Rayleigh wave phase velocity values [m/s].

    dc_ids : tuple
        Coordinates of dispersion curve data points (obtained by window 
        selection and/or by using spectral maxima IDs). 
        dc_ids[0] is a list of frequency values [Hz]. 
        dc_ids[1] is a list of Rayleigh wave phase velocity values [m/s].
    
    display_labels : tkinter.IntVar
        Value holder. Holds the control variable that describes the state of 
        the checkbutton to show/hide spectral maxima ID numbers. 

    draw_window : boolean
        Indicates the status of the selection window.

    elementdc : dispersion.ElementDC
        Elementary dispersion curve object.
    
    entered_ids : tkinter.StringVar
        Value holder. Holds entered/selected spectral maxima ID numbers. 

    f_min : int or float
        Lower limit of frequency axis [Hz].

    f_max : int or float
        Upper limit of frequency axis [Hz].

    ids_entry : tkinter.Entry
        Entry widget for selection of spectral maxima using point IDs.

    ids_labels : tkinter.Checkbutton
        Checkbutton to show/hide spectral maxima ID numbers.

    master : tkinter.Tk
        Parent widget. 

    msg : tkinter.Message
        Message widget to display multiline text in the footer frame 
        of the dispersion curve selection tool.

    point_ids : list of lists
        Selected spectral maxima identification (ID) numbers.

    rect : matplotlib.patches.Rectangle 
        Selection window.

    rect_end : tuple
        Coordinates of a selection window’s second diagonal point.

    rect_start : tuple
        Coordinates of a selection window’s first diagonal point.

    select_clicks : boolean
        Status of dispersion curve identification using mouse click events.

    select_ids : boolean
        Status of dispersion curve identification using window selection 
        and/or spectral maxima IDs

    toolbar : NavigationToolbar (matplotlib toolbar)
        Toolbar for the dispersion image.
        
    
    Instance methods
    ----------------
    btn_click_events_reset(self, frame)
        Mouse click events. Create a button to reset the selection of 
        dispersion curve data points.

    btn_click_events_start(self, frame)
        Mouse click events. Create a button to start selecting dispersion 
        curve data points.
    
    btn_click_events_stop(self, frame)
        Mouse click events. Create a button to stop the selection of 
        dispersion curve data points.
        
    btn_close_window(self, frame)
        Create a button to close the current window.

    btn_pointIDs_reset(self, frame)
        Window selection/selection using spectral maxima IDs. Create a button 
        to reset the dispersion curve identification.
        
    btn_pointIDs_start(self, frame)
        Window selection/selection using spectral maxima IDs. Create a button 
        to start the dispersion curve identification.
        
    btn_pointIDs_stop(self, frame)
        Window selection/selection using spectral maxima IDs. Create a button 
        to stop selection of spectral maxima.

    btn_save_selection(self, frame)
        Create a button to combine and sort dispersion curve data points 
        obtained using different identification methods.

    checkbtn_pointID_labels(self, frame)
        Create a check button to show/hide spectral maxima ID numbers.

    click_events_reset(self)
        Mouse click events. Clear current selection of dispersion curve 
        data points.

    click_events_start(self)
        Mouse click events. Pick dispersion curve data points.

    click_events_stop(self)
        Mouse click events. Stop selection of dispersion curve data points.

    configure_gui(self)
        Configure the parent (master) window.

    entry_pointIDs(self, frame)
        Window selection/selection using spectral maxima IDs. Create an entry 
        widget for entering spectral maxima ID numbers.

    message_to_user(self, message)
        Update the user instructions in the footer frame.

    pointIDs_reset(self)
        Window selection/selection using spectral maxima IDs. Clear current 
        selection of spectral maxima.

    pointIDs_start(self)
        Window selection/selection using spectral maxima IDs. Identify 
        dispersion curve from spectral maxima.

    pointIDs_stop(self)
        Window selection/selection using spectral maxima IDs. Stop selection 
        of spectral maxima.

    save_selection(self)
        Combine dispersion curve data points that are identified (i) based on 
        window selection/spectral maxima IDs and/or (ii) by clicking on 
        the dispersion image (mouse click events).

    select_pointIDs(self)
        Select spectral maxima by entering spectral maxima ID numbers 
        (point IDs) in an entry widget.

    selection_window(self)
        Select spectral maxima using a selection window.

    view_dispersion_image(self)
        Plot dispersion image and show the spectral maximum at each frequency.

    view_hide_labels(self)
        Show/hide spectral maxima ID numbers (point IDs).

    widgets_footer(self)
        Create footer frame and add widgets.

    widgets_header(self)
        Create header frame and add label.

    widgets_save_selection(self)
        Create widgets for combining and sorting dispersion curve data points 
        obtained using different identification methods and save to elementary 
        dispersion curve object.

    widgets_use_mouse_click_events(self)
        Create widgets for identifying (picking) dispersion curves by clicking 
        on the dispersion image.

    widgets_use_pointIDs(self)
        Add widgets for identifying (picking) dispersion curves using window 
        selection and/or spectral maxima ID numbers.

    _close_window(self)
        Close the GUI for dispersion curve selection.   

    _get_maxima(self)
        Find spectral maxima inside a selection window.

    _labels_to_str(self, n0)
        Reformat a list of selected point IDs as a comma-separated string. 

    _reset_selection_window(self)
        Reset the selection window.

    _str_to_labels(self, entered_ids)
        Reformat a comma-separated string of point IDs as a list of integers.

    
    Static methods
    -------------- 
    _get_rect(start, end)
        Obtain the anchor point, width, and height of a rectangle from 
        the start and end points of one of its diagonals.

    _in_rect(xp, yp, x, y, width, height)
        Check if a point is inside a rectangle.

    _remove_duplicates(list_of_ints)
        Remove duplicates from a list of integers. Return a sorted list of 
        unique elements.

    _split_list(list_of_ints)
        Split a list of positive integers into sublists. Each sublist 
        is sorted and does not include any missing values.
    
    """
        
    def __init__(self, ElementDC, f_min, f_max, master=None):
        
        """
        Initialize a dispersion curve selection object.
        
        Properties
        ----------
        ElementDC
            Elementary dispersion curve object.
        f_min : int or float
            Lower limit of frequency axis [Hz].
        f_max : int or float
            Upper limit of frequency axis [Hz].
        master
            Parent widget. 
            Default is master=None.
            
        Returns
        -------
        SelectDC
            Initialized dispersion curve selection object.
        
        """
        super().__init__(master)
        self.master = master
        
        self.elementdc = ElementDC
        self.f_min = f_min
        self.f_max = f_max
        
        # Initiate dispersion processing attributes
        self.dc_ids = None     
        self.point_ids = None         
        self.dc_clicks = None
             
        # Configure the parent window
        self.configure_gui()
        
        # Create widgets
        self.widgets_header()
        self.widgets_footer()
        self.widgets_use_pointIDs()  
        self.widgets_use_mouse_click_events()
        self.widgets_save_selection()
        
        # Plot dispersion image
        self.view_dispersion_image()
        
        # Status of start-selection buttons
        self.select_ids = False
        self.select_clicks = False
 
    
    def configure_gui(self):        
        
        """ 
        Configure the parent (master) window. 
        
        """
        self.master.title('MASWavesPy - Dispersion analysis')        
        self.master.geometry('{}x{}'.format(950, 750))
        self.master.grid_rowconfigure(7, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

   
    def widgets_header(self):        
        
        """ 
        Create header frame and add label. 
        
        """
        frm_header = tk.Frame(self.master, padx=9, pady=6)
        frm_header.grid(row=0, column=0, columnspan=4, sticky='w')

        lbl_header = tk.Label(frm_header, font=('DejaVu Sans', 12, 'bold'),
                                   text='Dispersion curve identification')
        lbl_header.grid(row=0, column=0, columnspan=4, sticky='w')


    def widgets_footer(self):
        
        """
        Create footer frame and add widgets.
        
        """
        frm_footer = tk.Frame(self.master, width=650, padx=9, pady=3)
        frm_footer.grid(row=10, column=0, columnspan=4, sticky='w')
          
        self.btn_close_window(frm_footer)
        self.checkbtn_pointID_labels(frm_footer)
        
        # Print user instructions
        temp_message = '\n \n \n \n \n'
        self.msg = tk.Message(frm_footer, width=650, text=temp_message, font=('DejaVu Sans', 9, 'normal')) 
        self.msg.grid(row=0, column=1, sticky='w')

        
    def btn_close_window(self, frame):
        
        """
        Create a button to close the current window.
        
        """
        btn_close = tk.Button(frame, text='Close', font=('DejaVu Sans', 10, 'bold'), 
                              foreground='black', height=1, width=7,
                              command=self._close_window)
        btn_close.grid(row=1, column=0, sticky='w', padx=3, pady=6)
    
    
    def _close_window(self):
        print("GUI for dispersion curve selection has been closed.")
        if self.elementdc.f0 is None:
            print("No dispersion curve data points have been saved. \n")        
        elif len(self.elementdc.f0) > 0:
            print(str(len(self.elementdc.f0)) + " dispersion curve data points have been selected and saved to the ElementDC object. \n")
        else:
            print("No dispersion curve data points have been saved. \n")
        self.master.destroy()
        
        
    def message_to_user(self, message):
        
        """
        Update user instructions in the footer frame. 
        
        Parameters
        ----------
        message : str
            Single-line or multi-line text message.
            
        """
        self.msg.configure(text=message)

# =============================================================================
#   Plot dipsersion image
# =============================================================================

    def view_dispersion_image(self):
        
        """
        Plot dispersion image and show the spectral maximum at each frequency.
        
        """
        # Create a figure and add axes
        fig = Figure(tight_layout=True)
        self.ax0 = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.12, top=0.92, left=0.11, right=0.99)
                
        # Plot dispersion image 
        self.elementdc.plot_dispersion_image(self.f_min, self.f_max, col_map='jet',  
                                             fig=fig, ax=self.ax0, tight_layout=False)
              
        # Plot spectral maxima
        Amax = self.elementdc.find_spectral_maxima()
        self.elementdc.Amax = Amax
        self.ax0.paths_maxima = []
        self.ax0.paths_maxima.append(self.elementdc.plot_spectral_maxima(Amax, ax=self.ax0, return_paths=True, edgecolor='k', color='k', point_size=5))
        self.ax0.paths_labels = []
        self.ax0.paths_ids = []
        
        # Add canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=7, sticky=tk.NSEW)

        # Add toolbar
        frm_toolbar = tk.Frame(self.master)
        frm_toolbar.grid(row=8, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, frm_toolbar)
        self.toolbar.update()
        
        # Update the dispersion image
        self.canvas.draw()
        
        # Print instructions
        start_message = ('Dispersion curve identification. \n'
            'The spectral maximum at each frequency is shown as a black dot on the dispersion image, each with its unique \n'
            'Point ID (spectral maxima id number). To show Point IDs, check the box at the bottom of the window. \n \n'
            'The dispersion curve is extracted by selecting the relevant spectral maxima using their Point IDs (recommended). Press START (Pick dispersion curve using Point IDs) to commence. \n \n'
            'Additional points can be added to the identified dispersion curve by clicking on the dispersion image. \n'
            'Press START (Pick dispersion curve by clicking on the image) to commence.')
        self.message_to_user(start_message)
 
    
# =============================================================================
#   Dispersion curve identification - window selection & point IDs
# =============================================================================   
    
    def _reset_selection_window(self):
        
        """
        Reset the selection window.
        
        """        
        while len(self.ax0.patches) > 0:
            self.ax0.patches[-1].remove()
        self.rect = Rectangle((0.0,0.0), 0.0, 0.0, fill=False, edgecolor='white', linewidth=1, linestyle='dashed')
        self.ax0.add_patch(self.rect)
        
        # Disable changes to the selection window
        self.draw_window = False   

    
    @staticmethod
    def _get_rect(start, end):
        
        """ 
        Obtain the anchor point (xy), width and height of a rectangle from 
        the start and end points of one of its diagonals.
        
        The rectangle extends from xy[0] to xy[0] + width in the x-direction 
        and from xy[1] to xy[1] + height in the y-direction.

        Properties
        ----------
        start : (float, float)
            Coordinates of the first diagonal point.
        end : (float, float)
            Coordinates of the second diagonal point.

        Returns
        -------
        xy : (float, float)
            Bottom and left rectangle coordinates.
        width : float
            Width of rectangle.
        height : float
            Height of rectangle.
            
        """
        xy = (min(start[0], end[0]), min(start[1], end[1]))  
        width = abs(end[0] - start[0])
        height = abs(end[1] - start[1])
        
        return xy, width, height
            

    @staticmethod
    def _in_rect(xp, yp, x, y, width, height):
        
        """
        Check if a point P = (xp, yp) is inside a rectangle.  
        
        The rectangle extends from x to x + width in the x-direction 
        and from y to y + height in the y-direction. 
        
        Properties
        ----------
        xp : float
            x-coordinate of point P.
        yp : float
            y-coordinate of point P.        
        x : float
            Bottom and left rectangle x-coordinate.
        y : float
            Bottom and left rectangle y-coordinate.
        width : float
            Width of rectangle.
        height : float
            Height of rectangle.
            
        Returns
        -------
        Boolean
            True if point P is inside the rectangle, else False.
            
        """      
        if x <= xp <= x + width and y <= yp <= y + height:
            return True
        else:
            return False


    def _get_maxima(self):
        
        """
        Find spectral maxima inside a selection window.
        
        Returns
        -------
        n0 : list of integers
            Selected spectral maxima, list of point IDs.        
            
        """     
        # Get selection window parameters
        xy, width, height = self._get_rect(self.rect_start, self.rect_end)
        
        # Find spectral maxima inside the selection window
        n0 = []
        for i in range(len(self.elementdc.Amax[0])):
            if self._in_rect(self.elementdc.Amax[0][i], self.elementdc.Amax[1][i], xy[0], xy[1], width, height):
                n0.append(i)
        
        return n0


    @staticmethod
    def _split_list(list_of_ints):
        
        """
        Split a list of positive integers into sublists. Each sublist is sorted 
        and does not include any missing values. 
        
        Example: 
        If list_of_ints = [1,2,3,7,9,10,11] then the returned list is
        main_list = [[1, 2, 3], [7], [9, 10, 11]]

        Parameters 
        ----------
        list_of_ints : list 
            List of integers.
            
        Returns
        -------
        main_list : list of lists
            List of sublists.
        
        """
        # Sort the list of integers
        list_of_ints.sort()
        
        # Split the list of integers into sublists.
        main_list = []
        sub_list = []
        prev_num = -1
        for num in list_of_ints:
            if prev_num + 1 != num:
                if sub_list:
                    main_list.append(sub_list)
                    sub_list = []
            sub_list.append(num)
            prev_num = num

        if sub_list:
            main_list.append(sub_list)

        return main_list
    

    def _labels_to_str(self, n0):
        
        """
        Reformat a list of selected point IDs as a comma-separated string. 
        If the list contains all IDs from x to y (inclusive), that selection 
        is returned as x-y.
        
        Example: 
        If n0 = [1,2,3,7,9,10,11] then the returned string is '1-3,7,9-11'.       
        
        Parameters 
        ----------
        n0 : list of integers
            Selected spectral maxima, list of point IDs.
        
        Returns
        -------
        string
            Comma-separated string of points IDs (formatted for the point ID 
            entry widget).
        
        """
        main_list = self._split_list(n0)
        
        temp_list = []
        for i in range(len(main_list)):
            if len(main_list[i]) <= 2:
                temp_list.append(','.join([str(j) for j in main_list[i]]))
            elif len(main_list[i]) > 2:
                temp_list.append('-'.join([str(main_list[i][j]) for j in (0, -1)]))  

        return ','.join([i for i in temp_list]) 


    def _str_to_labels(self, entered_ids):
            
        """
        Reformat a comma-separated string of point IDs as a list of integers.
        If the selected point IDs include all IDs from x to y (inclusive), 
        the selection may be entered as x-y.
        
        Example: 
        If entered_ids = '1-3,7,9-11' then the returned list is 
        n0 = [1,2,3,7,9,10,11].     
        
        Parameters 
        ----------
        entered_ids : str
            Comma-separated string of points IDs (formatted for the point ID 
            entry widget).
        
        Returns
        -------
        n0 : list of integers
            Selected spectral maxima, sorted list of point ID numbers.
        
        Raises
        ------
        ValueError
            If an entered ID number is outside the allowable range.
        
        """
        n0 = [int(ii) for ii in entered_ids.split(',') if ii.strip().isdigit()]
        
        for ii in range(len(entered_ids.split(','))):
            if '-' in entered_ids.split(',')[ii]:
                temp = [int(i) for i in entered_ids.split(',')[ii].split('-') if i.strip().isdigit()]
                n0 += list(range(temp[0], temp[1]+1))
    
        if n0:
            if min(n0) < 0 or max(n0) >= len(self.elementdc.Amax[0]):
                message = f'Entered Point IDs must be between 0 and {len(self.elementdc.Amax[0])-1} (inclusive).'
                raise ValueError(message)
    
        return sorted(n0)
    
    
    @staticmethod
    def _remove_duplicates(list_of_ints):
        
        """
        Remove duplicates from a list of integers. Return a sorted list of 
        unique elements.
        
        Parameters 
        ----------
        list_of_ints : list 
            List of integers.
            
        Returns
        -------
        list
            Sorted list of unique elements.
        
        """
        return sorted(list(set(list_of_ints)))
        
    
    def widgets_use_pointIDs(self):
        
        """ 
        Add widgets for identifying (picking) dispersion curves using 
        window selection and/or spectral maxima ID numbers.
        
        """
        frm_pointIDs = tk.Frame(self.master, background='gray85', 
                                highlightthickness=1, padx=3, pady=3)
        frm_pointIDs.grid(row=1, column=1, rowspan=2, columnspan=3, sticky='w')        
        lbl_pointIDs = tk.Label(frm_pointIDs, font=('DejaVu Sans', 10, 'bold'),
                                text='Pick dispersion curve \n using Point IDs',
                                background='gray85', borderwidth=5)
        lbl_pointIDs.grid(row=0, column=0, columnspan=3, padx=3, pady=3)
        
        frm_selectIDs = tk.Frame(self.master, padx=3, pady=3)
        frm_selectIDs.grid(row=9, column=0, sticky=tk.NSEW) 

        self.btn_pointIDs_start(frm_pointIDs)
        self.btn_pointIDs_reset(frm_pointIDs)       
        self.btn_pointIDs_stop(frm_pointIDs)  
        self.entry_pointIDs(frm_selectIDs)


    def checkbtn_pointID_labels(self, frame):
        
        """ 
        Create a check button to show/hide spectral maxima ID numbers
        (point IDs).
        
        Parameters
        ----------
        frame : tkinter.Frame
            
        """   
        self.display_labels = tk.IntVar()
        self.ids_labels = tk.Checkbutton(frame, text='Show Point IDs (spectral maxima ID numbers).', font=('DejaVu Sans', 9, 'bold'), 
                                         variable=self.display_labels, command=self.view_hide_labels)
        self.ids_labels.grid(row=1, column=1, sticky='w')
    
    
    def view_hide_labels(self):
        
        """
        Show/hide spectral maxima ID numbers (Point IDs).
        
        """
        # Show labels
        if self.display_labels.get() == 1:
            if not self.ax0.paths_labels:
                self.ax0.paths_labels += self.elementdc.label_spectral_maxima(self.elementdc.Amax, ax=self.ax0, return_text=True, 
                                                                             fontsize=8.5, ha='center', fontstretch='ultra-condensed')
        # Hide labels
        if self.display_labels.get() == 0:
            len_labels = len(self.ax0.paths_labels)
            if len_labels:
                for i in range(len_labels):
                    self.ax0.paths_labels.pop().remove()

        # Update the dispersion image
        self.canvas.draw() 


    def btn_pointIDs_start(self, frame):
        
        """ 
        Create a button to start selection using window selection and/or 
        spectral maxima IDs. 
        
        Parameters
        ----------
        frame : tkinter.Frame
            
        """
        ids_start = tk.Button(frame, text='Start', font=('DejaVu Sans', 10, 'bold'), 
                              foreground='black', height=1, width=6, 
                              command=self.pointIDs_start)
        ids_start.grid(row=1, column=0, sticky='w', padx=3, pady=3)


    def entry_pointIDs(self, frame):
        
        """
        Create an entry widget for selection of Point IDs (spectral maxima IDs).
        
        Parameters
        ----------
        frame : tkinter.Frame 
        
        """
        ids_entry_lbl = tk.Label(frame, font=('DejaVu Sans', 10, 'bold'),
                                text=' Point IDs:')
        ids_entry_lbl.grid(row=0, column=0, sticky='w', padx=3, pady=10)
        
        self.entered_ids = tk.StringVar()
        self.ids_entry = tk.Entry(frame, font=('DejaVu Sans', 11), width=75, textvariable=self.entered_ids, state='disabled')
        self.ids_entry.grid(row=0, column=1, sticky=tk.NSEW, padx=10, pady=10, ipady=3)

        ids_entry_scroll = tk.Scrollbar(frame, orient='horizontal', command=self.ids_entry.xview)
        ids_entry_scroll.grid(row=1, column=1, sticky='w')


    def selection_window(self):
        
        """
        Select spectral maxima using a selection window. 
        
        """
        # Initialize the window selection tool
        self.rect_start = (0.0, 0.0)
        self.rect_end = (0.0, 0.0)
                  
        while len(self.ax0.patches) > 0:
            self.ax0.patches[-1].remove()
        self.rect = Rectangle((0.0,0.0), 0.0, 0.0, fill=False, edgecolor='white', linewidth=1, linestyle='dashed')
        self.ax0.add_patch(self.rect)
        
        # Disable changes to the selection window
        self.draw_window = False

                   
        def on_btn_press(event):
            
            """
            Press left mouse button
            - Start drawing the selection window/rectangle.
            - Get the coordinates of its first diagonal point.
            
            Press right mouse button         
            - Deselect the last set of spectral maxima.
            - Remove the last set of spectral maxima from the dispersion image.
            - Update the point IDs entry widget.
                       
            """
            # Left mouse button: Start selecting points
            if event.inaxes is self.ax0 and event.button == 1:
                
                # Activate the selection window
                self.draw_window = True             
                
                # Get the coordinates of the first diagonal point
                self.rect_start = (event.xdata, event.ydata) 
            
            # Right mouse button: Deselect points
            if event.inaxes is self.ax0 and event.button == 3 and self.point_ids[-1]:
                
                # Update the list of selected spectral maxima
                self.point_ids.pop()
                n0_list = self.point_ids[-1]
                self.dc_ids = (self.elementdc.Amax[0][n0_list], self.elementdc.Amax[1][n0_list])  
                
                # Write to entry widget
                self.entered_ids.set(self._labels_to_str(n0_list))
                
                # Remove the last set of identified spectral maxima from dispersion image
                self.ax0.paths_ids.pop().remove()
                
                # Plot selected spectral maxima
                self.ax0.paths_ids = []
                self.ax0.paths_ids.append(self.ax0.scatter(self.dc_ids[0], self.dc_ids[1], s=11, color='white'))
                
                # Update the dispersion image
                self.canvas.draw()

        def on_pressing_p_key(event):
            
            """
            Press p (lower case) or P (upper case) on keyboard.
            - Pause selection of spectral maxima using the window selection tool.
            
            """
            if event.key =='p' or event.key == 'P':
                # Print instructions
                window_pause_message = ('Selection of Point IDs using the window selection tool has been paused. \n' 
                                        ' - Press START to resume. \n \n \n \n \n \n \n')
                self.message_to_user(window_pause_message)
                
                # Disconnect events
                self.canvas.mpl_disconnect(self.cid_window_start)
                self.canvas.mpl_disconnect(self.cid_window_update)
                self.canvas.mpl_disconnect(self.cid_window_quit)
                self.canvas.mpl_disconnect(self.cid_window_pause)
                
        def on_motion(event):
            
            """
            The left mouse button is pressed and the mouse moves 
            - Get the coordinates of the selection window's second diagonal point.
            - Draw the selection window.
            
            """
            if self.draw_window and event.inaxes is self.ax0:                   
                
                # Get the coordinates of the second diagonal point
                self.rect_end = (event.xdata, event.ydata)
                
                # Draw the selection window
                xy, width, height = self._get_rect(self.rect_start, self.rect_end)            
                self.rect.set_width(width)
                self.rect.set_height(height)
                self.rect.set_xy(xy)
                
                # Update the dispersion image
                self.canvas.draw()
                         
        def on_btn_release(event):            
            
            """
            Release left mouse button
            - Select spectral maxima inside a selection window.
            - Plot selected spectral maxima on top of dispersion image.
            - Get Point IDs and write to entry widget.
        
            """
            # Left mouse button: Select points
            if event.inaxes is self.ax0 and event.button == 1 and self.draw_window:
                
                # Find spectral maxima inside the current selection window (get point IDs).
                n0_window = self._get_maxima()
                
                if n0_window:                   
                    # Update the list of selected spectral maxima 
                    n0_list = self._remove_duplicates(self.point_ids[-1] + n0_window)
                    self.point_ids.append(n0_list)
                    
                    # Get coordinates of selected spectral maxima
                    self.dc_ids = (self.elementdc.Amax[0][n0_list], self.elementdc.Amax[1][n0_list])  
                    
                    # Write to entry widget
                    self.entered_ids.set(self._labels_to_str(n0_list))
               
                    # Remove the last set of identified spectral maxima from matplotlib plot
                    if self.ax0.paths_ids:
                        self.ax0.paths_ids.pop().remove()
               
                    # Plot selected spectral maxima
                    self.ax0.paths_ids = []
                    self.ax0.paths_ids.append(self.ax0.scatter(self.dc_ids[0], self.dc_ids[1], s=11, color='white'))
                    
                # Reset the selection window and update the dispersion image
                self._reset_selection_window()
                self.canvas.draw()

        # Receive events
        self.cid_window_start = self.canvas.mpl_connect('button_press_event', on_btn_press)
        self.cid_window_update = self.canvas.mpl_connect('motion_notify_event', on_motion)
        self.cid_window_quit = self.canvas.mpl_connect('button_release_event', on_btn_release)
        self.cid_window_pause = self.canvas.mpl_connect('key_press_event',on_pressing_p_key)
    
    
    def select_pointIDs(self):
        
        """
        Select spectral maxima by entering Point IDs (spectral maxima IDs)
        in an entry widget. 
        
        """          
        def on_return_key(event=None):

            """
            Press the enter/return key
            - Select the spectral maxima that correspond to the entered Point IDs.

            """
            # Get spectral maxima IDs (point IDs)
            entered_ids = self.entered_ids.get()
            
            # Update the list of selected spectral maxima
            n0_list =  self._remove_duplicates(self._str_to_labels(entered_ids))
            self.point_ids.append(n0_list)        
        
            # Get coordinates of selected spectral maxima
            self.dc_ids = (self.elementdc.Amax[0][n0_list], self.elementdc.Amax[1][n0_list])  
        
            # Remove the last set of identified spectral maxima from matplotlib plot
            if self.ax0.paths_ids:
                self.ax0.paths_ids.pop().remove()
        
            # Plot selected spectral maxima
            self.ax0.paths_ids = []
            self.ax0.paths_ids.append(self.ax0.scatter(self.dc_ids[0], self.dc_ids[1], s=11, color='white'))
            
            # Update the dispersion image
            self.canvas.draw()
        
 
        # Activate the entry widget  
        self.ids_entry.configure(state='normal')    
        
        # Bind to return key
        self.ids_entry.bind('<Return>', on_return_key)


    def pointIDs_start(self):
        
        """ 
        Identify experimental dispersion curves from spectral maxima. 
        
        Draw a selection window to select spectral maxima contained 
        within a rectangular area. 
        
        Enter spectral maxima IDs as a comma-separated list (e.g., 1,2,5,7,3,10). 
        To select all IDs from x to y (inclusive), the selection can be entered as x-y 
        (e.g., enter 1-3,7,9-11 to select Point IDs no. 1,2,3,7,9,10,11).
        
        """     
        # Deactivate dispersion curve selection using mouse click events (if already activated)
        if hasattr(self, 'cid_mouse'):
            self.canvas.mpl_disconnect(self.cid_mouse)
          
        # Print instructions
        ids_start_message = ('Window selection tool: \n'
            ' - Left-click at any point on the dispersion image and drag the cursor to include the maxima in the selection window.\n'
            ' - To pause the window selection (e.g., to zoom in/out), press P. \n'
            'Type Point IDs: \n'
            ' - Enter Point IDs as a comma-separated list (e.g., 1,2,5,7,3,10) and press the return/enter key.\n'
            ' - To select all IDs from x to y (inclusive), the selection can be entered as x-y (e.g., 1-3,7 to select IDs 1,2,3,7).\n'
            'Right-click at any point on the dispersion image to deselect the last set of spectral maxima. \n'
            'Selected points are shown as white dots. Press STOP to stop selecting Point IDs. Press RESET to reset the selection. \n')
        self.message_to_user(ids_start_message)

        # Label spectral maxima
        if not self.ax0.paths_labels:
            self.ax0.paths_labels += self.elementdc.label_spectral_maxima(self.elementdc.Amax, ax=self.ax0, return_text=True, 
                                                                             fontsize=8.5, ha='center', fontstretch='ultra-condensed')

        # Initiate dispersion curve identification using window selection 
        # and/or point IDs
        if self.select_ids is False:

            # Change the status of the start-selection button
            self.select_ids = True
                               
            # Initiate lists to store PathCollection objects and point IDs
            self.ax0.paths_ids = []
            self.point_ids = [[]]   
            
        # Activate the window selection
        self.selection_window()
        
        # Activate the point ID selection
        self.select_pointIDs()
        
        # Turn on checkbutton to label spectral maxima
        self.ids_labels.select()
    
        # Update the dispersion image
        self.canvas.draw()
                       
    
    def btn_pointIDs_reset(self, frame):
        
        """ 
        Create a button to reset the dispersion curve identification using  
        window selection and/or spectral maxima IDs.
        
        Parameters
        ----------
        frame : tkinter.Frame
            
        """       
        ids_reset = tk.Button(frame, text='Reset', font=('DejaVu Sans', 10, 'bold'), 
                              foreground='black', height=1, width=6,
                              command=self.pointIDs_reset)
        ids_reset.grid(row=1, column=1, sticky='w', padx=3, pady=3)
        

    def pointIDs_reset(self):

        """
        Clear current selection of spectral maxima.
        
        """
        # Print instructions
        ids_reset_message = ('Current selection of spectral maxima (Point IDs) has been cleared.\n '
            '- Press START to restart the dispersion curve identification. \n \n \n \n \n \n \n')            
        self.message_to_user(ids_reset_message)
        
        # Clear current selection and deactive the entry widget
        self.dc_ids = None
        if self.ids_entry.cget('state') == 'disabled':
            self.ids_entry.configure(state='normal')
        self.ids_entry.delete(0, 'end')
        self.ids_entry.configure(state='disabled')
        
        # Remove spectral maxima labels from matplotlib plot (dispersion image)
        len_labels = len(self.ax0.paths_labels)
        if len_labels:
            for i in range(len_labels):
                self.ax0.paths_labels.pop().remove()
                
        # Remove selected spectral maxima from matplotlib plot (dispersion image)
        if self.ax0.paths_ids:
            self.ax0.paths_ids.pop().remove()
        
        # Update the dispersion image
        self.canvas.draw() 
        
        # Change the status of the start-selection button
        self.select_ids = False
        
        # Disconnect events
        if hasattr(self, 'cid_window_start'):
            self.canvas.mpl_disconnect(self.cid_window_start)
        if hasattr(self, 'cid_window_update'):
            self.canvas.mpl_disconnect(self.cid_window_update)
        if hasattr(self, 'cid_window_quit'):
            self.canvas.mpl_disconnect(self.cid_window_quit)    
        self.ids_entry.unbind('<Return>')
        
        
    def btn_pointIDs_stop(self, frame):
        
        """ 
        Create a button to stop selection of spectral maxima.
        
        Parameters
        ----------
        frame : tkinter.Frame
               
        """
        ids_stop = tk.Button(frame, text='Stop', font=('DejaVu Sans', 10, 'bold'), 
                             foreground='black', height=1, width=6,
                             command=self.pointIDs_stop)
        ids_stop.grid(row=1, column=2, sticky='w', padx=3, pady=3)
      
    
    def pointIDs_stop(self):
        
        """
        Stop selection of spectral maxima.
                       
        """ 
        # Print instructions
        IDs_stop_message = ('The selection of spectral maxima is currently stopped. \n'
            ' - Press START to resume or alter the current selection. \n \n'
            'Press SAVE DISPERSION CURVE to complete the dispersion curve picking '
            'and view the identified experimental dispersion curve. \n \n \n \n')
        self.message_to_user(IDs_stop_message)
        
        # Deactive the entry widget
        self.ids_entry.configure(state='disabled')
        
        # Disconnect events
        if hasattr(self, 'cid_window_start'):
            self.canvas.mpl_disconnect(self.cid_window_start)
        if hasattr(self, 'cid_window_update'):
            self.canvas.mpl_disconnect(self.cid_window_update)
        if hasattr(self, 'cid_window_quit'):
            self.canvas.mpl_disconnect(self.cid_window_quit)    
        self.ids_entry.unbind('<Return>')
                
        
# =============================================================================
#   Dispersion curve identification - mouse click events  
# =============================================================================
    
    def widgets_use_mouse_click_events(self):

        """ 
        Create widgets for identifying (picking) dispersion curves by clicking
        on the dispersion image.
        
        """
        frm_clicks = tk.Frame(self.master, background='gray85', 
                              highlightthickness=1, padx=3, pady=3)
        frm_clicks.grid(row=3, column=1, rowspan=2, columnspan=3, sticky='w')
        lbl_clicks = tk.Label(frm_clicks, font=('DejaVu Sans', 10, 'bold'),
                              text=' Pick dispersion curve \n by clicking on the image ',
                              background='gray85', borderwidth=5)
        lbl_clicks.grid(row=0, column=0, columnspan=3, padx=3, pady=3)
        
        self.btn_click_events_start(frm_clicks)
        self.btn_click_events_reset(frm_clicks)       
        self.btn_click_events_stop(frm_clicks) 
    
    
    def btn_click_events_start(self, frame):
        
        """ 
        Create a button to start selecting dispersion curve data points 
        using mouse click events. 
        
        Parameters
        ----------
        frame : tkinter.Frame
            
        """
        clicks_start = tk.Button(frame, text='Start', font=('DejaVu Sans', 10, 'bold'), 
                                 foreground='black', height=1, width=6,
                                 command=self.click_events_start)
        clicks_start.grid(row=1, column=0, sticky='w', padx=3, pady=3)


    def click_events_start(self):
        
        """ 
        Pick dispersion curve using mouse click events. 
        
        """   
        # Initiate selection using mouse click events 
        if self.select_clicks is False:          
            
            # Change the status of the start-selection button
            self.select_clicks = True

            # Initiate lists to store coordinates of clicked points and
            # corresponding PathCollection objects
            self.dc_clicks = ([], [])
            self.ax0.paths_clicks = []    
         
        # Deactivate dispersion curve identification using window selection 
        # and/or point IDs (if already activated)
        if hasattr(self, 'cid_window_start'):
            self.canvas.mpl_disconnect(self.cid_window_start)
        if hasattr(self, 'cid_window_update'):
            self.canvas.mpl_disconnect(self.cid_window_update)    
        if hasattr(self, 'cid_window_quit'):
            self.canvas.mpl_disconnect(self.cid_window_quit)
        self.ids_entry.unbind('<Return>')
        
        # Deactivate the entry widget for selection using spectral maxima IDs         
        self.ids_entry.configure(state='disabled')  
        
        # Turn off checkbutton to label spectral maxima and remove labels
        len_labels = len(self.ax0.paths_labels)
        if len_labels:
            for i in range(len_labels):
                self.ax0.paths_labels.pop().remove()
        self.ids_labels.deselect()
        self.canvas.draw()
        
        # Print instructions
        clicks_start_message = ('Start selection of dispersion curve data points. \n' 
            ' - Use the left mouse button to select points. Selected points '
            'are indicated by white dots at the location of each click. \n'
            ' - Use the right mouse button to deselect the last selected point. \n'
            ' - Press P to pause the selection (e.g., to zoom in/out). \n'
            ' - Press STOP to stop the selection. \n'
            ' - Press RESET to reset the selection. \n \n \n')
        self.message_to_user(clicks_start_message)   
               
        def on_click(event):
            
            """ 
            Left mouse button:
            - Save x- and y-coordinates of selected points (add to lists). 
            - Plot selected points on top of dispersion image.
            
            Right mouse button:
            - Remove x- and y-coordinates of the last selected point from lists.
            - Remove the last selected point from the dispersion image.
           
            """            
            # Left mouse button: Select points
            if event.inaxes is self.ax0 and event.button == 1:
                self.dc_clicks[0].append(event.xdata)
                self.dc_clicks[1].append(event.ydata)
                self.ax0.paths_clicks.append(self.ax0.scatter(event.xdata, event.ydata, s=11, color='white'))
                self.canvas.draw()
                             
            # Right mouse button: Deselect points
            if event.inaxes is self.ax0 and event.button == 3:
                if self.ax0.paths_clicks:
                    self.dc_clicks[0].pop()
                    self.dc_clicks[1].pop()
                    self.ax0.paths_clicks.pop().remove()
                    self.canvas.draw()
                        
        def on_p_key(event):
            
            """
            Press p (lower case) or P (upper case) on keyboard:
            - Pause the selection of dispersion curve data points.
            
            """            
            if event.key =='p' or event.key == 'P':
                # Print instructions
                clicks_pause_message = ('Selection of dispersion curve data points has been paused. \n' 
                                        ' - Press START to resume. \n \n \n \n \n \n \n')
                self.message_to_user(clicks_pause_message)
                
                if hasattr(self, 'cid_mouse'):
                    self.canvas.mpl_disconnect(self.cid_mouse)
                    self.canvas.mpl_disconnect(self.cid_p_key)                   
        
        self.cid_mouse = self.canvas.mpl_connect('button_press_event', on_click)
        self.cid_p_key = self.canvas.mpl_connect('key_press_event',on_p_key)

            
    def btn_click_events_reset(self, frame):
        
        """ 
        Create a button to reset the selection of dispersion curve data points 
        (using mouse click events). 
        
        Parameters
        ----------
        frame : tkinter.Frame
            
        """    
        clicks_reset = tk.Button(frame, text='Reset', font=('DejaVu Sans', 10, 'bold'), 
                                 foreground='black', height=1, width=6,
                                 command=self.click_events_reset)
        clicks_reset.grid(row=1, column=1, sticky='w', padx=3, pady=3)


    def click_events_reset(self):
        
        """
        Clear current selection of dispersion curve data points (using 
        mouse click events).
        
        """
        # Print instructions
        clicks_reset_message = ('Current selection of disperson curve data points has been cleared. \n'
            '- Press START to restart the dispersion curve identification. \n'
            '\n \n \n \n \n \n')            
        self.message_to_user(clicks_reset_message)
        
        # Remove selected points from matplotlib plot (dispersion image)
        if self.ax0.paths_clicks:
            n_points = len(self.ax0.paths_clicks)
            for i in range(n_points):
                self.ax0.paths_clicks.pop().remove()
        
        # Clear current selection
        self.dc_clicks = None     
        
        # Update the dispersion image
        self.canvas.draw()
       
        # Pause selection of spectral maxima
        if hasattr(self, 'cid_mouse'):
            self.canvas.mpl_disconnect(self.cid_mouse)
        
        # Change the status of the start-selection button
        self.select_clicks = False
        
              
    def btn_click_events_stop(self, frame):
        
        """ 
        Create a button to stop the selection of dispersion curve data points
        (using mouse click events).
        
        Parameters
        ----------
        frame : tkinter.Frame
               
        """      
        clicks_stop = tk.Button(frame, text='Stop', font=('DejaVu Sans', 10, 'bold'),
                                foreground='black', height=1, width=6,
                                command=self.click_events_stop)
        clicks_stop.grid(row=1, column=2, sticky='w', padx=3, pady=3)


    def click_events_stop(self):
        
        """
        Stop selection of dispersion curve data points using mouse click
        events. 
                       
        """                  
        # Print instructions
        clicks_stop_message = ('The selection of dispersion curve data points is currently stopped. \n'
            ' - Press START to select additional points or to alter the currenct selection. \n \n'
            'Press SAVE DISPERSION CURVE to complete the dispersion curve picking '
            'and view the identified experimental dispersion curve. \n \n \n \n')
        self.message_to_user(clicks_stop_message)
        
        # Update the dispersion image
        self.canvas.draw()

        # Pause selection of spectral maxima
        if hasattr(self, 'cid_mouse'):
            self.canvas.mpl_disconnect(self.cid_mouse)


# =============================================================================
#   Dispersion curve identification - save selection
# =============================================================================

    def widgets_save_selection(self):
        
        """
        Create widgets for combining and sorting dispersion curve data points  
        obtained using different identification methods and save to the 
        corresponding elementary dispersion curve object (elementdc).
        
        """
        frm_save_selection = tk.Frame(self.master, background='gray85',
                                      highlightthickness=1, padx=3, pady=3)
        frm_save_selection.grid(row=5, column=1, rowspan=2, columnspan=3, sticky='w')
        lbl_save_selection = tk.Label(frm_save_selection, font=('DejaVu Sans', 10, 'bold'),
                                      text=' Complete selection of \n dispersion curve ',
                                      background='gray85', borderwidth=5)
        lbl_save_selection.grid(row=0, column=0, columnspan=3, padx=3, pady=3)
        
        self.btn_save_selection(frm_save_selection)
    
    
    def btn_save_selection(self, frame):
        
        """
        Create a button to combine and sort dispersion curve data points  
        obtained using different identification methods. 
        
        """
        btn_save = tk.Button(frame, text='Save dispersion curve', font=('DejaVu Sans', 10, 'bold'), 
                             foreground='black', height=1, width=22,
                             command=self.save_selection)
        btn_save.grid(row=1, column=0, columnspan=3, sticky='w', padx=3, pady=3)
    
    
    def save_selection(self):
    
        """
        Combine dispersion curve data points that are identified (i) based on 
        window selection/spectral maxima IDs and/or (ii) by clicking on 
        the dispersion image for a given elementdc object.
        
        The experimental dispersion curve is returned as numpy arrays 
        elementdc.f0_vec (frequency [Hz]) and c0_vec (Rayleigh wave 
        phase velocity [m/s]). The data points are sorted by frequency values 
        in order of increasing frequency. 
        
        """  
        # Print instructions
        save_message = ('The selected dispersion curve has been saved to the elementary dispersion curve object. \n' 
             'The saved dispersion curve data points are indicated by black dots. \n \n'
             'Press CLOSE to close the dispersion curve identification window. \n \n \n \n \n')
        self.message_to_user(save_message)       
        
        # Data points identified using window selection and/or based on spectral maxima IDs
        if not (self.dc_ids is None): 
            f0_temp = self.dc_ids[0]
            c0_temp = self.dc_ids[1]
        else:
            f0_temp = np.empty(0,)
            c0_temp = np.empty(0,)
            
        # Data points identified by clicking on the dispersion image
        if not (self.dc_clicks is None): 
            f0_temp = np.concatenate((f0_temp, self.dc_clicks[0]))
            c0_temp = np.concatenate((c0_temp, self.dc_clicks[1]))
            
        # Remove duplicates, sort in order of increasing frequency and 
        # save to elementary dispersion curve object
        dc_temp = np.unique(np.array((f0_temp,c0_temp)), axis=1)
        self.elementdc.f0 = dc_temp[0]
        self.elementdc.c0 = dc_temp[1]
        
        # Turn off checkbutton to label spectral maxima and remove labels
        len_labels = len(self.ax0.paths_labels)
        if len_labels:
            for i in range(len_labels):
                self.ax0.paths_labels.pop().remove()
        self.ids_labels.deselect()
        
        # Remove spectral maxima from matplotlib plot (dispersion image)
        if self.ax0.paths_maxima:
            self.ax0.paths_maxima.pop().remove()
        
        # Plot saved data points on top of an existing dispersion image
        self.ax0.scatter(self.elementdc.f0, self.elementdc.c0, s=11, color='black')
        
        # Update the dispersion image
        self.canvas.draw()
