# This is a sample Python script.
import csv
import pandas as pd
import pyarrow
import fastparquet
import names
from random import randrange
from collections import OrderedDict
from faker import Faker
from datetime import datetime, timezone

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(' in main: generating names ')
    locales = OrderedDict([
        ('en-US', 1)
    ])
    fake = Faker(locales)

    customerFields = ['CUSTOMER_ID', 'FIRSTNAME', 'LASTNAME', 'DOB', 'PEP']
    emailFields = ['EMAIL_ID', 'CUSTOMER_ID', 'EMAIL']
    ssnFields = ['SSN_ID', 'CUSTOMER_ID', 'SSN']
    addressFields = ['ADDRESS_ID', 'CUSTOMER_ID', 'BUILDING_NUMBER', 'STREET_NAME', 'CITY', 'COUNTRY', 'POSTCODE']
    #
    customerData = []
    emailData = []
    ssnData = []
    addressData = []
    pepCount = 0
    for i in range(0, 999):
        customerDataEntry = dict()
        customerDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        fullName = fake.name()
        customerDataEntry['FIRSTNAME'] = fullName.split(' ')[0]
        customerDataEntry['LASTNAME'] = fullName.split(' ')[1]
        customerDataEntry['DOB'] = fake.date_of_birth(tzinfo=timezone.utc , minimum_age = 16, maximum_age = 65)
        customerDataEntry['PEP'] = 'false'
        if randrange(1000) < 4:
            print("pep")
            pepCount = pepCount + 1
            customerDataEntry['PEP'] = 'true'
        customerData.append(customerDataEntry)
        #
        emailDataEntry = dict()
        emailDataEntry['EMAIL_ID'] = 'EMAIL_' + str(i)
        emailDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        emailDataEntry['EMAIL'] = fake.company_email()
        emailData.append(emailDataEntry)
        #
        ssnDataEntry = dict()
        ssnDataEntry['SSN_ID'] = 'SSN_' + str(i)
        ssnDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        ssnDataEntry['SSN'] = fake.ssn()
        ssnData.append(ssnDataEntry)
        #
        addressDataEntry = dict()
        addressDataEntry['ADDRESS_ID'] = 'ADDRESS_' + str(i)
        addressDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        addressDataEntry['BUILDING_NUMBER'] = fake.building_number()
        addressDataEntry['STREET_NAME'] = fake.street_name()
        addressDataEntry['CITY'] = fake.city()
        addressDataEntry['COUNTRY'] = fake.country()
        addressDataEntry['POSTCODE'] = fake.postcode()
        addressData.append(addressDataEntry)

    for i in range(0, 1):
        customerDataEntry = dict()
        customerDataEntry['CUSTOMER_ID'] = 'S_' + str(i+1000)
        fullName = 'CALLE SERNA'
        customerDataEntry['FIRSTNAME'] = fullName.split(' ')[0]
        customerDataEntry['LASTNAME'] = fullName.split(' ')[1]
        customerDataEntry['DOB'] = fake.date_of_birth(tzinfo=timezone.utc, minimum_age=16, maximum_age=65)
        customerDataEntry['PEP'] = 'false'
        customerData.append(customerDataEntry)
        #
        emailDataEntry = dict()
        emailDataEntry['EMAIL_ID'] = 'EMAIL_' + str(i)
        emailDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        emailDataEntry['EMAIL'] = fake.company_email()
        emailData.append(emailDataEntry)
        #
        ssnDataEntry = dict()
        ssnDataEntry['SSN_ID'] = 'SSN_' + str(i)
        ssnDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        ssnDataEntry['SSN'] = fake.ssn()
        ssnData.append(ssnDataEntry)
        #
        addressDataEntry = dict()
        addressDataEntry['ADDRESS_ID'] = 'ADDRESS_' + str(i)
        addressDataEntry['CUSTOMER_ID'] = 'C_' + str(i)
        addressDataEntry['BUILDING_NUMBER'] = fake.building_number()
        addressDataEntry['STREET_NAME'] = fake.street_name()
        addressDataEntry['CITY'] = fake.city()
        addressDataEntry['COUNTRY'] = fake.country()
        addressDataEntry['POSTCODE'] = fake.postcode()
        addressData.append(addressDataEntry)

    # writing to names csv file
    with open('names.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=customerFields, delimiter=',')
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(customerData)
    print(pepCount)
    print(' wrote customer data ')
    # writing to email csv file
    with open('email.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writerEmail = csv.DictWriter(csvfile, fieldnames=emailFields, delimiter=',')
        # writing headers (field names)
        writerEmail.writeheader()
        # writing data rows
        writerEmail.writerows(emailData)
    print(' wrote email data ')
    # writing to ssn csv file
    with open('ssn.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=ssnFields, delimiter=',')
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(ssnData)
    print(' wrote ssn data ')
    # writing to address csv file
    with open('address.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=addressFields, delimiter=',')
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(addressData)
    print(' wrote address data ')