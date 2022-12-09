# This is a sample Python script.
import csv
from collections import OrderedDict
from datetime import timezone

import numpy as numpy
from faker import Faker
import pandas as pd
import pyarrow
import fastparquet
import names
from random import randrange

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(' in main: generating extended transaction attributes ')
    locales = OrderedDict([
        ('en-US', 1)
    ])
    fake = Faker(locales)

    fields = ['TX_ID', 'method', 'suspectRegion', 'date']
    dataList = []
    pepCount = 0

    ##
    batchSize=10000
    with open(file):
        for line in file:
            batchOfLines.append(line)
            if len(batchOfLines)>batchSize:
                for textLine in batchOfLines:
                    d(like a dictionary)=parseCSV(textLine)
                    # access to d['TX_ID'] is available
                    extendStruct to add additional fields (method, etc.)
                    #d['method'] = 'online'
                    #d['suspectRegion'] = 'false'
                    #d['date'] = fake.date_of_birth(tzinfo=timezone.utc, minimum_age=1, maximum_age=3)
                    #
                    writeToFile in append mode
                    alternatively
                    write a new file for each batch using DictWriter
            #remember to clear the batchOfLines array (assign a new array)

    ##




    for i in range(1, 117533):
        d = dict()
        rDevice = numpy.random.normal(5, 3)
        rRegion = numpy.random.normal(5, 3)
        d['TX_ID'] = str(i)
        d['method'] = 'online'
        d['suspectRegion'] = 'false'
        d['date'] = fake.date_of_birth(tzinfo=timezone.utc, minimum_age=1, maximum_age=3)
        if rDevice < 2:
            d['method'] = 'atm'
        elif rDevice > 8:
            d['method'] = 'branch'
        #
        if rRegion < 0:
            pepCount = pepCount + 1
            d['suspectRegion'] = 'true'

        dataList.append(d)

    # writing to csv file
    with open('tx_extension.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(dataList)
    print(pepCount)
    print(' wrote dict ')
