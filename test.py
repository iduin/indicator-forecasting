# Create_graph_multi.py reads a excel file and creates graphs for the data in the file
# The graphs are saved in the same directory as the excel file
# The graphs are saved in the format of .png

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pdb

def draw_one(data, min_size, max_size, pred, graph_size):
    """
    Draw data in the serie
    :param data: the list of values (list) initial data
    :param min_size: the minimum size of the data to draw
    :param max_size: the maximum size of the data to draw
    :param pred: the number of values to predict in percentage
    :param graph_size: the size of the graph (number of points)
    :return: the drawn serie
    """
    # Draw a size with an uniform distribution between min_size and max_size
    size = np.random.randint(min_size, max_size)
    # Get a random start index between 0 and the size of the values - size
    start = np.random.randint(0, len(data[0]) - size)
    # Get the data to draw
    new_data = []
    for serie in data:
        serie = serie[start:start + size]
        # Resample the data to the graph_size + Pred (%) using linear interpolation
        si = (np.round((1.0 + pred / 100) * graph_size)).astype(int)
        serie = np.interp(np.linspace(0, 1, si), np.linspace(0, 1, len(serie)), serie)
        new_data.append(serie)
    return new_data

######
def plot_one(data, names, size = 256):
    """
    draw a serie using matplotlib
    :param values: the values of the serie
    :name: name of the values
    :return: image of the serie
    """
    fig = plt.figure()
    # plot in black and white (background is black)
    plt.style.use('dark_background')
    # plot the values in white
    for values, name in zip(data, names):
        # Normalize the values between 0 and 1
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        plt.plot(values, label=name)
    # fig size in pixel is size pixels
    fig.set_size_inches(size/100, size/100)
    # remove the axis
    plt.axis('off')
    plt.style.use('dark_background')
    # No border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
   

def create_graphs(file_path, nb_graphs_per_thousand = 300, min_size = 50, max_size = 300, pred = 10, graph_size = 256):
    # Read the excel file
    data = pd.ExcelFile(file_path)
    #data = pd.read_excel(file_path)
    # get all sheet names
    sheet_names = data.sheet_names
    sheet_test = data.parse(sheet_names[0])
    # Loop on all the sheets
    for sheet_name in sheet_names:
        sheet = data.parse(sheet_name)
        print(f'Processing sheet {sheet_name}')
        # if the sheet is empty, skip it
        if sheet.empty:
            continue
        # if the sheet is sheet_test, skip it
        #if sheet.equals(sheet_test):
        #    continue
        # List of values to extract
        value_names = ['Open', 'High', 'Low', 'Volume','Close','Close Lissé (Lag<=2)', 'Stoch (15,15,1) TS', 'MACD (12,26,1)',
       'RSI (14)', 'ADX (14)']
        lvalues = []
        for value_name in value_names:
            if value_name in sheet.columns:
                tmp = sheet[value_name].values
                # remove 60 first values (stochl)
                tmp = tmp[60:]
                lvalues.append(tmp)
            else:
                print(f'Value {value_name} not found in sheet {sheet_name}')

        # get the size of the values
        size = len(lvalues[0])
        # Compute the number of graphs to create
        nb_graphs = np.round(size * nb_graphs_per_thousand // 1000)
        # Loop
        for i in range(nb_graphs):
            # draw one serie
            serie = draw_one(lvalues, min_size, max_size, pred, graph_size)
            #
            # plot the serie (visible part)
            # new list with graph_size values

            serie_crop = []
            for serie_ in serie:
                serie_crop.append(serie_[:graph_size])
            fig = plot_one(serie_crop, value_names)
            # Assume that close is the -2 serie
            # find index for value to predict (tendancy)
            index = value_names.index('RSI (14)')
            pre = serie[index][graph_size:]
            # Compute tendancy of pred
            tendancy = pre[-1] - pre[0]
            # if tendancy is positive, save in positive folder else in negative folder 
            if tendancy > 0:
                folder = 'data/positive/'
            else:
                folder = 'data/negative/'
            # Save the figure
            fig.savefig(folder + sheet_name + '_' + str(i) + '.png', dpi=100)
            plt.close(fig)
            # Save serie in a excel file
            df = pd.DataFrame(serie)
            # in a xlsx
            df.to_excel(folder + sheet_name + '_' + str(i) + '.xlsx', index=False)

# Main
if __name__ == "__main__":
    # Create graphs for the data in the excel file
    create_graphs("Base_Test_2500pts.xlsx")
