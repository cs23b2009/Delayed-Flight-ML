import pandas as pd

# Suppose your mapping is in a DataFrame
# with columns: IATA, ICAO, USAF
mapping_df = pd.read_csv("iata-noaa.csv")

# List of IATA codes you care about
wanted_iata = ['ABY', 'ATL', 'MOB', 'BUF', 'BTV', 'CVG', 'LGA', 'CHO', 'EWN', 'MCI', 'MGM', 'MSP', 'DCA', 'FAY', 'OAJ', 'STL', 'CWA', 'DTW', 'RDU', 'SRQ', 'AEX', 'AUS', 'CSG', 'EVV', 'IND', 'CLE', 'PIA', 'TRI', 'BGR', 'IAD', 'PWM', 'TYS', 'GSO', 'AVL', 'BTR', 'MKE', 'CMH', 'GRR', 'GSP', 'BIS', 'ORD', 'ROC', 'CHA', 'FSM', 'JAX', 'BOS', 'GRB', 'TUL', 'VLD', 'CRW', 'IAH', 'XNA', 'JAN', 'CHS', 'ROA', 'GFK', 'SYR', 'MDT', 'SAT', 'BMI', 'LFT', 'BNA', 'PHL', 'BQK', 'ORF', 'GNV', 'DFW', 'OMA', 'TPA', 'SAV', 'BWI', 'MEM', 'MBS', 'LEX', 'CLT', 'DAY', 'MOT', 'FAR', 'CAE', 'DSM', 'PIT', 'DHN', 'MHT', 'ILM', 'SGF', 'ELM', 'BDL', 'MCO', 'RIC', 'SHV', 'MSY', 'HSV', 'TLH', 'LAN', 'LIT', 'GRK', 'MSN', 'ATW', 'HRL', 'HPN', 'CAK', 'EWR', 'RSW', 'BQN', 'PBI', 'FLL', 'SLC', 'PSE', 'JFK', 'SJC', 'LAS', 'LGB', 'LAX', 'SFO', 'SJU', 'HOU', 'DEN', 'SEA', 'STT', 'PVD', 'PDX', 'SAN', 'RNO', 'ALB', 'PHX', 'OAK', 'PSP', 'SMF', 'STX', 'BUR', 'SWF', 'DAB', 'ORH', 'ABQ', 'ELP', 'LCH', 'HOB', 'PNS', 'COS', 'ICT', 'CLL', 'OKC', 'ECP', 'MFE', 'MYR', 'BHM', 'AGS', 'AZO', 'GPT', 'RST', 'VPS', 'EYW', 'PHF', 'ABE', 'CID', 'LNK', 'MLI', 'SFB', 'BIL', 'FNT', 'MSO', 'IDA', 'SCK', 'TOL', 'FSD', 'HTS', 'FAT', 'EUG', 'BLI', 'PGD', 'STC', 'PIE', 'PBG', 'SBN', 'OGS', 'USA', 'SPI', 'PVU', 'AZA', 'MTJ', 'FWA', 'SDF', 'PSM', 'TTN', 'IAG', 'LCK', 'BLV', 'RFD', 'OGD', 'BOI', 'MFR', 'GTF', 'PSC', 'GRI', 'MRY', 'GJT', 'LRD', 'RAP', 'FCA', 'SMX', 'BZN', 'HGR', 'CKB', 'OWB', 'HNL', 'OGG', 'KOA', 'LIH', 'ITO', 'ACY', 'ONT', 'ACV', 'BFL', 'ASE', 'TWF', 'GEG', 'RDM', 'TUS', 'DRO', 'SBA', 'RDD', 'COU', 'SNA', 'MMH', 'JAC', 'CPR', 'SBP', 'MAF', 'ISN', 'STS', 'BRO', 'HDN', 'ITH', 'LWS', 'PIH', 'ABR', 'APN', 'ESC', 'PLN', 'BJI', 'BRD', 'BTM', 'CDC', 'SGU', 'CIU', 'EKO', 'COD', 'HIB', 'BGM', 'MQT', 'RHI', 'LSE', 'INL', 'DAL', 'TVC', 'GTR', 'SCE', 'ERI', 'IMT', 'DLH', 'MDW', 'HLN', 'SUN', 'AVP', 'MLU', 'CMX', 'EAU', 'GCC', 'RKS', 'PUB', 'MKG', 'CGI', 'UIN', 'DVL', 'JMS', 'LAR', 'HYS', 'PAH', 'EGE', 'OTH', 'AMA', 'ISP', 'MIA', 'PPG', 'LBE', 'CRP', 'LBB', 'MLB', 'ANC', 'FAI', 'GUM', 'GUC', 'ADQ', 'BET', 'SCC', 'BRW', 'JNU', 'KTN', 'SIT', 'OME', 'OTZ', 'YAK', 'CDV', 'WRG', 'PSG', 'ADK', 'SAF', 'DIK', 'CMI', 'SPN', 'ROP', 'PUW', 'EAT', 'ALW', 'YKM', 'YNG', 'DUT', 'BFF', 'ACK', 'MVY', 'BKG', 'TYR', 'SJT', 'BPT', 'SWO', 'LAW', 'ACT', 'FLG', 'YUM', 'ROW', 'MEI', 'PIB', 'LBF', 'LWB', 'SHD', 'EAR', 'SLN', 'VEL', 'CNY', 'LBL', 'PRC', 'MHK', 'ABI', 'GCK', 'PGV', 'HVN', 'LYH', 'PQI', 'HHH', 'GGG', 'SUX', 'TXK', 'DBQ', 'SBY', 'FLO', 'IPT', 'ART', 'ALO', 'SPS', 'JLN', 'CYS', 'DRT', 'LNY', 'MKK', 'JHM', 'HYA', 'GST', 'AKN', 'DLG', 'WYS']

print(len(wanted_iata))

for i in wanted_iata:
    if i not in list(mapping_df['iata']):
        print(i)
# Filter mapping to only keep rows where IATA is in that list
# filtered = mapping_df[mapping_df['iata'].isin(wanted_iata)]

# print(filtered.shape)   # should be ~370 rows
# print(filtered.head())

# # Optionally save back to Excel/CSV
# filtered.to_csv("filtered_mapping.csv", index=False)
