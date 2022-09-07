

# #################################################################


#  This program accepts no arguments, and outputs a number of CSV files.
#
#  .  See 'SCALE_FACTOR" as the only real tunable-
#
#  .  Maybe run this program inside a new folder, to avoid sprawl of
#     newly created files.
#
#  .  This program reads at least one CSV file, and the call we use here
#     may not work on GS/S3.


#  Need Faker installed
#     https://faker.readthedocs.io/en/master/locales/en_US.html#faker-providers-date-time
#
#  pip install Faker 


import os

try:
   from faker import Faker
except ImportError:
   os.system("pip3 install Faker")
   from faker import Faker

from collections import OrderedDict
# from datetime import datetime, timezone
from datetime import datetime
   #
import csv
   #
from random import randint




# #################################################################
# #################################################################


#  SCALE_FACTOR 5000, generates around 15M rows
#
SCALE_FACTOR = 50


#  SCALE_FACTOR,
#
#  .  Agent             are made 1:1     (SCALE_FACTOR 5 ==  5   Agent(s))
#                                          --
#                                        And each Agent has 1 Location
#
#  .  Attorney          are made 1:1     (SCALE_FACTOR 5 ==  5   Attorney(s))
#                                          --
#                                        And each Attorney has 1 Location
#
#  .  Provider          are made 10:1    (SCALE_FACTOR 5 ==  50  Provider(s))
#                                          --
#                                        And each Provider has 1 Location
#
#  .  Claimant          are made 100:1   (SCALE_FACTOR 5 ==  500 Claimant(s))
#                                          --
#                                        And each Claimant has 1 Location
#
#  **  Locations above are generated randomly, and as such, are not shared 
#      between any of the nodes above.
#
#
#  **  Below this point the data model diagram gets a little hinky, which
#      we will accept, because in the real world, data would be incomplete.
#      To explain;
#
#      --  Policy is drawn as relating to Claims, and not Claimant.
#          If there is no Claim against a Policy, then there is no 
#          relationship between the Claimant and the Policy; a ghost
#          Policy, if you will.
#
#      --  Payments are drawn as relating to Claimant, when they could
#          best relate to a Claim.
#
#      --  MBRs (medical billing records) are drawn as relating to
#          Policy and Claim, when the relationship to Claim is really
#          derived.
#
#
#  .  Policy  (and MBRs)                 Are drawn in the data model as relating
#                                        to,
#                                        -- Agents
#                                           and are made 100:1
#                                           that is, each Agent has 100 Policies
#
#                                        -- MBRs  (medical billing records)
#                                           and are made 0-3:1
#                                           that is, each Policy has 0-3 MBRs
#                                           generated randomly
#                                              #
#                                           Each MBR is randomly assigned an 
#                                           existing Provider
#                                           (Each MBR is given a Location, following
#                                           the Location comment above; that is, a
#                                           net new random Location.)
#
#                                       **  Currently, we generate MBR at the time
#                                           we generate Policy. This is before the
#                                           time we have generated a Claim.
#
#                                           So, we leave CLAIM_ID in the MBR NULL.
#
#                                           MBRs relate to Policy, and Claim will
#                                           relate to Policy; perhaps a neat part 
#                                           to the demo ?  (Matching Claims ?)
#
#                                       **  Per the data model diagram, an MBR was
#                                           supposed to have a Location. We already
#                                           have 5 or more random Locations, so I 
#                                           left Location off from MBR.
#
#
#  .  Claim                             --  Relates to Policy, Attorney, and Claimant
#
#                                           Attorney is set to SCALE_FACTOR
#                                           Claimant is set to SCALE_FACTOR * 100
#                                           Policy   is set to SCALE_FACTOR + 1, * 1-100
# 
#                                           Since Policy has a random, we'll read Policy,
#                                           and output Claims 0-5 relative to Policy
#
#  .  Payment                           --  Was drawn in the data model diagram as relating
#                                           to Claimant and Location. Since Claimant already
#                                           has a Location and Locations are random, we only
#                                           relate Payment to Claimant.
#                                          
#                                           Claimant is created SCALE_FACTOR * 100
#                                           So, we'll create Payment at random 0-1.




# #################################################################


l_locales = OrderedDict([
   ('en-US', 1)
   ])

l_faker_locales = Faker(l_locales)


edge_fields              = ['START_ID', 'END_ID']

agent_fields             = ['ID', 'LABEL', 'FIRSTNAME', 'LASTNAME', 'DOB', 'EMAIL', 'SSN']
attorney_fields          = ['ID', 'LABEL', 'FIRSTNAME', 'LASTNAME', 'DOB', 'EMAIL', 'SSN']
provider_fields          = ['ID', 'LABEL', 'FIRSTNAME', 'LASTNAME', 'DOB', 'EMAIL', 'SSN', 'LICENSE_NUMBER']

# claimant_fields          = ['ID', 'LABEL', 'FIRSTNAME', 'LASTNAME', 'DOB', 'EMAIL', 'SSN']
claimant_fields          = ['ID', 'LABEL', 'FIRSTNAME', 'LASTNAME', 'DOB', 'EMAIL', 'SSN', 'CITY', 'COUNTRY', 'POSTALCODE_PLUS4', 'STATE_ABBR', 'STREET_ADDRESS']

location_fields          = ['ID', 'LABEL', 'CITY', 'COUNTRY', 'POSTALCODE_PLUS4', 'STATE_ABBR', 'STREET_ADDRESS']

policy_fields            = ['ID', 'LABEL', 'POLICY_NUMBER', 'POLICY_BIND_DATE', 'POLICY_STATE', 'POLICY_CSL', 
                            'POLICY_DEDUCTABLE', 'POLICY_ANNUAL_PREMIUM', 'UMBRELLA_LIMIT', 'AUTO_MAKE', 'AUTO_YEAR',
                            'AGENT_ID', 'CLAIM_ID']

mbr_fields               = ['ID', 'LABEL', 'POLICY_ID', 'CLAIM_ID', 'PROVIDER_ID', 'AMOUNT', 'LICENSE_NUMBER',
                            'INSURED_SEX', 'INSURED_EDUCATION_LEVEL', 'INSURED_OCCUPATION', 'INSURED_HOBBIES',
                            'INSURED_RELATIONSHIP', 'INCIDENT_DATE', 'INCIDENT_TYPE', 'INCIDENT_CITY',
                            'INCIDENT_STATE', 'INCIDENT_SEVERITY', 'AUTHORITIES_CONTACTED', 'POLICE_REPORT_AVAILABLE']

claim_fields             = [ 'ID', 'LABEL', 'POLICY_ID', 'CLAIMANT_ID', 'ATTORNEY_ID', 'AMOUNT', 'CLAIM_DATE', 
                             'CAPITAL_GAINS', 'CAPITAL_LOSS', 'WITNESSES']

payment_fields           = ['ID', 'LABEL', 'PAYMENT_DATE', 'AMOUNT', 'CURRENCY']




# #################################################################
# #################################################################


#  1.) Just Agent, Agent_Location, Location
#
#  Yes, this write to memory, then to file, does not scale super well.

def create_agent_to_csv(i_cntr, i_outputfile1, i_outputfile2, i_outputfile3):


   agents          = []
   locations       = []
   agent_locations = []

   
   #  Loop to create agents array
   #
   print("")
   print("")
   print("Step 1: Agent, Agent Location, Location")
   for l_cntr in range(0, i_cntr):
   
      agent = dict()
         #
      agent['ID']                    = "A_" + str(l_cntr)
      agent['LABEL']                 = "agent"   
         #
      l_name                         = l_faker_locales.name()
      agent['FIRSTNAME']             = l_name.split(' ')[0]
      agent['LASTNAME']              = l_name.split(' ')[1]
         #
      # agent['DOB']                   = l_faker_locales.date_of_birth(tzinfo=timezone.utc , minimum_age = 21, maximum_age = 90)
      agent['DOB']                   = l_faker_locales.date_of_birth(minimum_age = 21, maximum_age = 90)
      agent['EMAIL']                 = l_faker_locales.company_email()
      agent['SSN']                   = l_faker_locales.ssn()
         #
      agents.append(agent)

            ###

      location = dict()
         #
      location['ID']                 = "AL_" + str(l_cntr)
      location['LABEL']              = "agent_location"
         #
      location['CITY']               = l_faker_locales.city()
#     location['COUNTRY']            = l_faker_locales.country()
      location['COUNTRY']            = "USA"
      location['POSTALCODE_PLUS4']   = l_faker_locales.postalcode_plus4()
      location['STATE_ABBR']         = l_faker_locales.state_abbr()
      location['STREET_ADDRESS']     = l_faker_locales.street_address()
         #
      locations.append(location)
      
            ###

      agent_location1 = dict()
         #
      agent_location1['START_ID']     = "A_"  + str(l_cntr)
      agent_location1['END_ID']       = "AL_" + str(l_cntr)
         #
      agent_locations.append(agent_location1)
         
#     agent_location2 = dict()
#        #
#     agent_location2['START_ID']     = "AL_" + str(l_cntr)
#     agent_location2['END_ID']       = "A_"  + str(l_cntr)
#        #
#     agent_locations.append(agent_location2)


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 3")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=agent_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(agents)
   
   print("   Flushing: 2 of 3")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=location_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(locations)
   
   print("   Flushing: 3 of 3")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(agent_locations)
   



# #################################################################


#  2.) Just Attorney, Attorney_Location, Location


def create_attorney_to_csv(i_cntr, i_outputfile1, i_outputfile2, i_outputfile3):


   attorneys          = []
   locations          = []
   attorney_locations = []

   
   #  Loop to create attorneys array
   #
   print("Step 2: Attorney, Attorney Location, Location")
   for l_cntr in range(0, i_cntr):
   
      attorney = dict()
         #
      attorney['ID']                    = "X_" + str(l_cntr)
      attorney['LABEL']                 = "attorney"
         #
      l_name                            = l_faker_locales.name()
      attorney['FIRSTNAME']             = l_name.split(' ')[0]
      attorney['LASTNAME']              = l_name.split(' ')[1]
         #
      # attorney['DOB']                   = l_faker_locales.date_of_birth(tzinfo=timezone.utc , minimum_age = 21, maximum_age = 90)
      attorney['DOB']                   = l_faker_locales.date_of_birth(minimum_age = 21, maximum_age = 90)
      attorney['EMAIL']                 = l_faker_locales.company_email()
      attorney['SSN']                   = l_faker_locales.ssn()
         #
      attorneys.append(attorney)

            ###

      location = dict()
         #
      location['ID']                    = "XL_" + str(l_cntr)
      location['LABEL']                 = "attorney_location"
         #
      location['CITY']                  = l_faker_locales.city()
#     location['COUNTRY']               = l_faker_locales.country()
      location['COUNTRY']               = "USA"
      location['POSTALCODE_PLUS4']      = l_faker_locales.postalcode_plus4()
      location['STATE_ABBR']            = l_faker_locales.state_abbr()
      location['STREET_ADDRESS']        = l_faker_locales.street_address()
         #
      locations.append(location)
      
            ###

      attorney_location1 = dict()
         #
      attorney_location1['START_ID']     = "X_"  + str(l_cntr)
      attorney_location1['END_ID']       = "XL_" + str(l_cntr)
         #
      attorney_locations.append(attorney_location1)
   
#     attorney_location2 = dict()
#        #
#     attorney_location2['START_ID']     = "XL_" + str(l_cntr)
#     attorney_location2['END_ID']       = "X_"  + str(l_cntr)
#        #
#     attorney_locations.append(attorney_location2)


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 3")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=attorney_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(attorneys)

   print("   Flushing: 2 of 3")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=location_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(locations)

   print("   Flushing: 3 of 3")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(attorney_locations)




# #################################################################


#  3.) Just Provider (Doctor), Provider_Location, Location


def create_provider_to_csv(i_cntr, i_outputfile1, i_outputfile2, i_outputfile3):


   providers          = []
   locations          = []
   provider_locations = []

   
   #  Loop to create providers array
   #
   print("Step 3: Provider (Doctor), Provider Location, Location")
   for l_cntr in range(0, i_cntr):
   
      provider = dict()
         #
      provider['ID']                    = "P_" + str(l_cntr)
      provider['LABEL']                 = "provider"
         #
      l_name                            = l_faker_locales.name()
      provider['FIRSTNAME']             = l_name.split(' ')[0]
      provider['LASTNAME']              = l_name.split(' ')[1]
         #
      # provider['DOB']                   = l_faker_locales.date_of_birth(tzinfo=timezone.utc , minimum_age = 21, maximum_age = 90)
      provider['DOB']                   = l_faker_locales.date_of_birth(minimum_age = 21, maximum_age = 90)
      provider['EMAIL']                 = l_faker_locales.company_email()
      provider['SSN']                   = l_faker_locales.ssn()
         #
      provider['LICENSE_NUMBER']        = l_faker_locales.license_plate()          # Yes, I know, wrong type of license   :)
         #
      providers.append(provider)

            ###

      location = dict()
         #
      location['ID']                    = "PL_" + str(l_cntr)
      location['LABEL']                 = "provider_location"
         #
      location['CITY']                  = l_faker_locales.city()
#     location['COUNTRY']               = l_faker_locales.country()
      location['COUNTRY']               = "USA"
      location['POSTALCODE_PLUS4']      = l_faker_locales.postalcode_plus4()
      location['STATE_ABBR']            = l_faker_locales.state_abbr()
      location['STREET_ADDRESS']        = l_faker_locales.street_address()
         #
      locations.append(location)
      
            ###

      provider_location1 = dict()
         #
      provider_location1['START_ID']     = "P_"  + str(l_cntr)
      provider_location1['END_ID']       = "PL_" + str(l_cntr)
         #
      provider_locations.append(provider_location1)
  
#     provider_location2 = dict()
#        #
#     provider_location2['START_ID']     = "PL_" + str(l_cntr)
#     provider_location2['END_ID']       = "P_"  + str(l_cntr)
#        #
#     provider_locations.append(provider_location2)


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 3")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=provider_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(providers)

   print("   Flushing: 2 of 3")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=location_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(locations)

   print("   Flushing: 3 of 3")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(provider_locations)




# #################################################################


#  4.) Just Claimant (a Person), Claimant_Location, Location


def create_claimant_to_csv(i_cntr, i_outputfile1, i_outputfile2, i_outputfile3):


   claimants          = []
   locations          = []
   claimant_locations = []

   
   #  Loop to create claimant array
   #
   print("Step 4: Claimant, Claimant Location, Location")
   for l_cntr in range(0, i_cntr):
   
      claimant = dict()
         #
      claimant['ID']                    = "C_" + str(l_cntr)
      claimant['LABEL']                 = "claimant"
         #
      l_name                            = l_faker_locales.name()
      claimant['FIRSTNAME']             = l_name.split(' ')[0]
      claimant['LASTNAME']              = l_name.split(' ')[1]
         #
      # claimant['DOB']                   = l_faker_locales.date_of_birth(tzinfo=timezone.utc , minimum_age = 21, maximum_age = 90)
      claimant['DOB']                   = l_faker_locales.date_of_birth(minimum_age = 21, maximum_age = 90)
      claimant['EMAIL']                 = l_faker_locales.company_email()
      claimant['SSN']                   = l_faker_locales.ssn()
         #
      claimant['CITY']                  = l_faker_locales.city()
#     claimant['COUNTRY']               = l_faker_locales.country()
      claimant['COUNTRY']               = "USA"
      claimant['POSTALCODE_PLUS4']      = l_faker_locales.postalcode_plus4()
      claimant['STATE_ABBR']            = l_faker_locales.state_abbr()
      claimant['STREET_ADDRESS']        = l_faker_locales.street_address()
         #
      claimants.append(claimant)

            ###

      location = dict()
         #
      location['ID']                    = "CL_" + str(l_cntr)
      location['LABEL']                 = "claimant_location"
         #
      location['CITY']                  = l_faker_locales.city()
#     location['COUNTRY']               = l_faker_locales.country()
      location['COUNTRY']               = "USA"
      location['POSTALCODE_PLUS4']      = l_faker_locales.postalcode_plus4()
      location['STATE_ABBR']            = l_faker_locales.state_abbr()
      location['STREET_ADDRESS']        = l_faker_locales.street_address()
         #
      locations.append(location)
      
            ###

      claimant_location1 = dict()
         #
      claimant_location1['START_ID']     = "C_"  + str(l_cntr)
      claimant_location1['END_ID']       = "CL_" + str(l_cntr)
         #
      claimant_locations.append(claimant_location1)

#     claimant_location2 = dict()
#        #
#     claimant_location2['START_ID']     = "CL_" + str(l_cntr)
#     claimant_location2['END_ID']       = "C_"  + str(l_cntr)
#        #
#     claimant_locations.append(claimant_location2)


   #  Write to CSV
   #
   print("   Flushing: 1 of 3")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=claimant_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claimants)

   print("   Flushing: 2 of 3")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=location_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(locations)

   print("   Flushing: 3 of 3")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claimant_locations)




# #################################################################
# #################################################################


#  5.) Just Policy, MBR, and associated edges


def create_policy_to_csv(i_cntr, i_outputfile1, i_outputfile2, i_outputfile3, i_outputfile4):


   auto_makes          = ['jeep', 'porsche', 'audi', 'ford', 'chevy', 'rivian', 'canoo', 'volvo']
      #
   policys             = []               #  pardon the spelling mistake
   policys_agents      = []
      #
   mbrs                = [] 
   mbrs_providers      = [] 

   
   #  Loop to create policy array, mbrs, and associated edges
   #
   print("Step 5: Policy, MBR, (and associated Edges)")
   for l_cntr1 in range(0, i_cntr):
      for l_cntr2 in range(1, 100):

         #  Just Policy in this block
         #
         policy                          = dict()
            #
         policy['ID']                    = "Q_" + str(l_cntr1) + "." + str(l_cntr2)
         policy['LABEL']                 = "policy"
            #
         policy['POLICY_NUMBER']         = "Q_" + str(l_cntr1) + "." + str(l_cntr2)
            #
         policy['POLICY_BIND_DATE']      = l_faker_locales.date_between(datetime.strptime("20210101", "%Y%m%d"))
         policy['POLICY_STATE']          = l_faker_locales.state_abbr()
         policy['POLICY_CSL']            = randint(10000  , 100000    ) / 100
         policy['POLICY_DEDUCTABLE']     = randint(1000   , 10000     ) / 100
         policy['POLICY_ANNUAL_PREMIUM'] = randint(10000  , 100000    ) / 100
         policy['UMBRELLA_LIMIT']        = randint(1000000, 1000000000) / 100
            #
         policy['AUTO_MAKE']             = auto_makes[randint(0,7)]
            #
         policy['AUTO_YEAR']             = l_faker_locales.year()
         #
         #  Agent already exists, on a scale factor equal to l_cntr1 above
         #
         policy['AGENT_ID']              = "A_" + str(l_cntr1)
            #
         policys.append(policy)


         #  Creating the edge between Policy and Agent
         #
         policy_agent1                   = dict()
         policy_agent2                   = dict()
            #
         policy_agent1['START_ID']       = policy['ID'      ]
         policy_agent1['END_ID'  ]       = policy['AGENT_ID']
            #
         policys_agents.append(policy_agent1)

#        policy_agent2['START_ID']       = policy['AGENT_ID']
#        policy_agent2['END_ID'  ]       = policy['ID'      ]
#           #
#        policys_agents.append(policy_agent2)


         #  Now MBR, in this block
         #
         for l_cntr3 in range(0, randint(0, 3)):

            mbr = dict()
               #
            mbr['ID']                          = "M_" + str(l_cntr1) + "." + str(l_cntr2) + "." + str(l_cntr3)
            mbr['LABEL']                       = "mbr"
               #
            mbr['POLICY_ID']                   = "Q_" + str(l_cntr1) + "." + str(l_cntr2)

            mbr['CLAIM_ID']                    = ""


            #
            #  Provider already exists, on a scale factor equal to l_cntr1 above, * 10
            #
            #  With the random(), and more, some Providers might not have an MBR.
            #  Also, some of the records in the generated edge might be duplicates, 
            #  meaning; you might see 6000 lines in the CSV, but only 4000 edges
            #  actually create
            #
            # MMM
            # mbr['PROVIDER_ID']                 = "P_" + str(l_cntr1) + "." + str(randint(1, 10))
            mbr['PROVIDER_ID']                 = "P_" + str(randint(0, SCALE_FACTOR * 10))


               #
            mbr['AMOUNT']                      = randint(10000  , 100000    ) / 100
               #
            mbr['LICENSE_NUMBER']              = l_faker_locales.license_plate()          # Yes, I know, wrong type of license   :)
               #
            mbr['INSURED_SEX']                 = randint(0, 3 )
            mbr['INSURED_EDUCATION_LEVEL']     = ""
            mbr['INSURED_OCCUPATION']          = ""
            mbr['INSURED_HOBBIES']             = ""
            mbr['INSURED_RELATIONSHIP']        = ""
            mbr['INCIDENT_DATE']               = l_faker_locales.date_between(datetime.strptime("20210101", "%Y%m%d"))
            mbr['INCIDENT_TYPE']               = randint(0, 13)
            mbr['INCIDENT_CITY']               = l_faker_locales.city()
            mbr['INCIDENT_STATE']              = l_faker_locales.state_abbr()
            mbr['INCIDENT_SEVERITY']           = randint(0, 6 )
            mbr['AUTHORITIES_CONTACTED']       = randint(0, 1 )
            mbr['POLICE_REPORT_AVAILABLE']     = randint(0, 1 )
               #
            mbrs.append(mbr)

            #  Creating the edge between MBR and Provider
            #
            mbr_provider1                   = dict()
            mbr_provider2                   = dict()
               #
            mbr_provider1['START_ID']       = mbr['ID'         ]
            mbr_provider1['END_ID']         = mbr['PROVIDER_ID']
               #
            mbrs_providers.append(mbr_provider1)
#              #
#           mbr_provider2['START_ID']       = mbr['PROVIDER_ID']
#           mbr_provider2['END_ID']         = mbr['ID'         ]
#              #
#           mbrs_providers.append(mbr_provider2)


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 4")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=policy_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(policys)

   print("   Flushing: 2 of 4")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(policys_agents)

   print("   Flushing: 3 of 4")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=mbr_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(mbrs)

   print("   Flushing: 4 of 4")
   with open (i_outputfile4, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(mbrs_providers)




# #################################################################
# #################################################################


#  6.) Claims, and edges to Attorney, Policy and Claimant


def create_claim_to_csv(SCALE_FACTOR, i_outputfile1, i_outputfile2,
      i_outputfile3, i_outputfile4, i_inputfile):


   claims             = []
   claims_attorneys   = []
   claims_policys     = []
   claims_claimants   = []

   #  Since Policy had a random(), we'll read that file, and output
   #  Claims random 0-5


   l_file = open(i_inputfile)
   l_csvreader = csv.reader(l_file, delimiter="|")
   l_header    = next(l_csvreader)
      #
   l_cntr1 = 1
      #
   print("Step 6: Claims, (and associated Edges to Attorney, Policy, Claimant)")
   for l_policy in l_csvreader:
      l_cntr1     = l_cntr1 + 1
      l_policy_id = l_policy[0]
         #
      for l_cntr2 in range(0, randint(0, 5)):
         claim   = dict()
            #
   
         # MMM
         claim['ID']                = "D_" + str(l_cntr1) + "." + str(l_cntr2)
         claim['LABEL']             = "claim"
            #
         claim['POLICY_ID']         = l_policy_id
         claim['CLAIMANT_ID']       = "C_" + str(randint(0, SCALE_FACTOR) * 100)
         claim['ATTORNEY_ID']       = "X_" + str(randint(0, SCALE_FACTOR)      )
            #
         claim['AMOUNT']            = randint(10000  , 100000    ) / 100
         claim['CLAIM_DATE']        = l_faker_locales.date_between(datetime.strptime("20210101", "%Y%m%d"))
         claim['CAPITAL_GAINS']     = randint(100    , 100000    ) / 100
         claim['CAPITAL_LOSS']      = randint(100    , 100000    ) / 100
         claim['WITNESSES']         = randint(0, 5)
            #
         claims.append(claim)


         # Create the edges to each of; Attorney, Policy, Claimant
         #
         claim_attorney1                   = dict()
         claim_attorney2                   = dict()
            #
         claim_attorney1['START_ID']       = claim['ID'         ]
         claim_attorney1['END_ID']         = claim['ATTORNEY_ID']
            #
         claims_attorneys.append(claim_attorney1)
#           #
#        claim_attorney2['START_ID']       = claim['ATTORNEY_ID']
#        claim_attorney2['END_ID']         = claim['ID'         ]
#           #
#        claims_attorneys.append(claim_attorney2)


         claim_policy1                   = dict()
         claim_policy2                   = dict()
            #
         claim_policy1['START_ID']       = claim['ID'       ]
         claim_policy1['END_ID']         = claim['POLICY_ID']
            #
         claims_policys.append(claim_policy1)
#           #
#        claim_policy2['START_ID']       = claim['POLICY_ID']
#        claim_policy2['END_ID']         = claim['ID'       ]
#           #
#        claims_policys.append(claim_policy2)


         claim_claimant1                   = dict()
         claim_claimant2                   = dict()
            #
         claim_claimant1['START_ID']       = claim['ID'         ]
         claim_claimant1['END_ID']         = claim['CLAIMANT_ID']
            #
         claims_claimants.append(claim_claimant1)
#           #
#        claim_claimant2['START_ID']       = claim['CLAIMANT_ID']
#        claim_claimant2['END_ID']         = claim['ID'         ]
#           #
#        claims_claimants.append(claim_claimant2)


   l_file.close()                                    #  The reading of Policys, above


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 4")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=claim_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claims)

   print("   Flushing: 2 of 4")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claims_attorneys)

   print("   Flushing: 3 of 4")
   with open (i_outputfile3, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claims_policys)

   print("   Flushing: 4 of 4")
   with open (i_outputfile4, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(claims_claimants)




# #################################################################
# #################################################################


#  7.) Payments, and edge to Claimant


def create_payment_to_csv(i_cntr, i_outputfile1, i_outputfile2):


   payments           = []
   payments_claimants = []

   
   #  Loop to create payment array
   #
   print("Step 7: Payments, (and associated Edge to Claimant)")
   for l_cntr in range(0, i_cntr):
   
      for _ in range(0, randint(0, 1)):

         payment = dict()
            #
         payment['ID']                     = "P_"  + str(l_cntr)
         payment['LABEL']                  = "payment"
            #
         payment['PAYMENT_DATE']           = l_faker_locales.date_between(datetime.strptime("20210101", "%Y%m%d"))
         payment['AMOUNT']                 = randint(100, 100000) / 100
         payment['CURRENCY']               = "dollar"
            #
         payments.append(payment)

         payment_claimant1 = dict()
         payment_claimant1['START_ID']     = "P_"  + str(l_cntr)
         payment_claimant1['END_ID']       = "C_"  + str(l_cntr)
            #
         payments_claimants.append(payment_claimant1)

#        payment_claimant2 = dict()
#        payment_claimant2['START_ID']     = "C_"  + str(l_cntr)
#        payment_claimant2['END_ID']       = "P_"  + str(l_cntr)
#           #
#        payments_claimants.append(payment_claimant2)


   #  Write to CSV(s)
   #
   print("   Flushing: 1 of 2")
   with open (i_outputfile1, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=payment_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(payments)

   print("   Flushing: 2 of 2")
   with open (i_outputfile2, "w") as csvfile:
      l_writer = csv.DictWriter(csvfile, fieldnames=edge_fields, delimiter="|")
         #
      l_writer.writeheader()
      l_writer.writerows(payments_claimants)




# #################################################################
# #################################################################


if __name__ == '__main__':

   #  This program is only tested/designed to accept change in SCALE_FACTOR
   #
   create_agent_to_csv(   SCALE_FACTOR      , "v_agent.csv"        , "v_agent_location.csv"   , "e_agent_location.csv"   )
   create_attorney_to_csv(SCALE_FACTOR      , "v_attorney.csv"     , "v_attorney_location.csv", "e_attorney_location.csv")
   create_provider_to_csv(SCALE_FACTOR * 10 , "v_provider.csv"     , "v_provider_location.csv", "e_provider_location.csv")
   create_claimant_to_csv(SCALE_FACTOR * 100, "v_claimant.csv"     , "v_claimant_location.csv", "e_claimant_location.csv")

   create_policy_to_csv(  SCALE_FACTOR      , "v_policy.csv"       , "e_policy_agent.csv"     ,
                                              "v_mbr.csv"          , "e_mbr_provider.csv"     )

   create_claim_to_csv(   SCALE_FACTOR      , "v_claim.csv"        , "e_claim_attorney.csv"   ,
                                              "e_claim_policy.csv" , "e_claim_claimant.csv"   ,
                                              "v_policy.csv"                                  )

   create_payment_to_csv( SCALE_FACTOR * 100, "v_payment.csv"      , "e_payment_claimant.csv" )





