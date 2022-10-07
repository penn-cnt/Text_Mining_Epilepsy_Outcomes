import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import string
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import transformers
        
class Patient:
    """Patient object. Contains information about the patient's identifiers, visits and medications"""
    def __init__(self, pat_id):
        self.pat_id = pat_id
        self.visits = []
        self.medications = []
        
    def add_visit(self, visit):
        self.visits.append(visit)
        
    def add_medication(self, medication):
        self.medications.append(medication)
        
    def __eq__(self, other):
        if isinstance(other, Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
    def __str__(self):
        return self.pat_id
    
    
class Aggregate_Patient:
    """Aggregate Patient Object. Contains information about a patient's identifiers, medications, and aggregate visits."""
    def __init__(self, pat_id):
        self.pat_id = pat_id
        self.aggregate_visits = []
        self.medications = []
        
    def add_aggregate_visit(self, aggregate_visit):
        self.aggregate_visits.append(aggregate_visit)
        
    def add_medication(self, Medication):
        self.medications.append(Medication)
        
    def __eq__(self, other):
        if isinstance(other, Aggregate_Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
    def __str__(self):
        return self.pat_id
    
    
class Visit:
    """
    Visit Object. Generated from information from a single medical note
        Patient: The Patient with this Visit
        note_id: The visit's ID
        author: The name of the provider
        visit_date: The date of the visit
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
        context: The note text of the visit
    """
    def __init__(self, patient, note_id, author, visit_date, hasSz, pqf, elo, context):
        
        #visit identifiers
        self.Patient = patient
        self.note_id = note_id
        self.author = author
        self.visit_date = visit_date
        
        #the patient's seizure freedom classification, seizure frequency, and date of last seizure, respectively
        self.hasSz = hasSz
        self.pqf = pqf
        self.elo = elo
        
        #the visit's note text
        self.context = context
        
        self.time_to_first_visit = -1 #how much time has passed since the patient's first visit?
        
        #automatically add this visit to the Patient's list
        if self not in self.Patient.visits:
            self.Patient.add_visit(self)
        else:
            print("Warning: This visit already exists for this patient. Visit was not added to Patient's visit list")
        
    def __str__(self):
        """Prints information for this visit"""
        return f"Visit for patient {self.Patient.pat_id} on {self.visit_date}, written by {self.author}: HasSz = {self.hasSz}; pqf_per_month = {self.pqf}; elo = {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Visit):
            return (self.Patient == other.Patient) and (self.note_id == other.note_id) and (self.visit_date == other.visit_date) and (self.author == other.author)
        else:
            return False
        
        
class Aggregate_Visit:
    """
    Class for a visit that combines multiple of the same visit (if multiple models with different seeds make predictions for the same visit)
        Aggregate_Patient: The Patient with this Visit
        pat_id: The patient identifier associated with the Aggregate Patient
        note_id: The visit's ID
        author: The name of the provider
        visit_date: The date of the visit
        all_visits: A list of the (same) visits that make up this single aggregate visit. 
        context: The note text of the visit
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
    """
    def __init__(self, note_id, visit_date, author, all_visits, aggregate_patient=None, pat_id=None):
        #first, check if the visits are all the same
        if all_visits.count(all_visits[0]) != len(all_visits):
            raise ValueError(f"Not all visits are the same")
            
        #next, check if a patient identifier was specified
        if pd.isnull(aggregate_patient) and pd.isnull(pat_id):
            raise ValueError(f"No patient identifier found")
            
        #if a pat_id and Aggregate_Patient was supplied, then check if they match identifiers
        if not pd.isnull(aggregate_patient) and not pd.isnull(pat_id):
            if aggregate_patient.pat_id != pat_id:
                raise ValueError(f"Aggregate_Patient pat_id and passed pat_id do not match")
            
        #get the basic info for the visit
        self.Aggregate_Patient = aggregate_patient
        self.all_visits = all_visits
        self.pat_id = pat_id if not pd.isnull(pat_id) else self.Aggregate_Patient.pat_id
        self.note_id = note_id
        self.visit_date = visit_date
        self.author = author
        self.time_to_first_visit = -1 #how much time has passed since the patient's first visit?
        
        #get the hasSz, pqf and elo for each visit
        self.all_hasSz = [vis.hasSz for vis in all_visits]
        self.all_pqf = [vis.pqf if not (pd.isnull(vis.pqf) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an placemarker arbitrary value for aggregate functions (below)
        self.all_elo = [vis.elo if not (pd.isnull(vis.elo) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an arbitrary placemarker value for aggregate functions (below)
        
        #calculate plurality voting
        self.hasSz = self.get_aggregate_hasSz()
        self.pqf = self.get_aggregate_pqf()
        self.elo = self.get_aggregate_elo()
        
        if self.Aggregate_Patient:
            #automatically add this aggregate visit to the Aggregate_Patient's list
            if self not in self.Aggregate_Patient.aggregate_visits:
                self.Aggregate_Patient.add_aggregate_visit(self)
            else:
                print("Warning: This visit already exists for this patient. Visit was not added to Aggregate_Patient's visit list")

        
    def get_aggregate_hasSz(self):
        """ 
        Gets the seizure freedom value for the aggregate visit by (plurality) voting.
        If there is a tie at the highest number of votes,
            If yes and no have the same number of votes, then default to IDK
            If Yes or No has the same number votes as IDK, then default to either Yes or No
        """        
        #count the votes
        votes = dict.fromkeys(set(self.all_hasSz), 0)
        for vote in self.all_hasSz:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0]
        #otherwise, if 0,1 both have the highest number of visits, then return idk (2)
        elif (0 in most_vals) and (1 in most_vals):
            return 2
        #otherwise, it must be that either 0 and 1 are tied with idk (2). Return either the 0 or 1
        else:
            most_vals.sort() #sort, since IDK is always 2
            return most_vals[0]
        
    def get_aggregate_pqf(self):
        """
        Calculate the seizure frequency with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_pqf), 0)
        for vote in self.all_pqf:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def get_aggregate_elo(self):
        """
        Calculate the date of last seizure with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_elo), 0)
        for vote in self.all_elo:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def __str__(self):
        return f"Aggregate Visit Object for {self.pat_id} on {self.visit_date}, written by {self.author}. hasSz: {self.hasSz}, pqf: {self.pqf}, elo: {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Aggregate_Visit):
            return (self.pat_id == other.pat_id) and (self.visit_date == other.visit_date) and (self.author == other.author) and (self.all_visits == other.all_visits)
        else:
            return False
        
#https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
def sort_by_visit_date(x):
    """Sorts a series/list of visits by their visit_dates"""
    return [i for (i, vis) in sorted(enumerate(x), key=lambda x:x[1].visit_date)]


def process_calendar(unprocessed_text):
    """
    Detects if a string is a seizure calendar by checking if:
        it involves at least two months
        there are at least 2 numbers of seizures
        the associated number of seizures is always 1 word-index afterwards, or within 3 word spaces postwards.
    Input:
        unprocessed_text: a string - generally the output of the finetuned seizure frequency extraction model
    Returns:
        None if it is not a seizure calendar,
        or
        The seizure frequency as a float, with units "per month", if it is a seizure calendar.
    """
    #the abbreviations for months
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    #remove punctuation and split the text by spaces 
    #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    text = unprocessed_text.replace('.', ' . ').replace(',', ' , ').replace('-', ' - ').translate(str.maketrans('','', string.punctuation)).split()
    
    #what are the months in the text, from 1-12?
    month_idx = []

    #this dictionary maps a month idx/position in the text to its associated number idx/position
    month_to_number = {}
    #this dictionary maps a number idx/position in the text to its associated month idx/position
    number_to_month = {}

    #for each word in text, check if it has a number, or a month
    for j in range(len(text)):
        if any(char.isdigit() for char in text[j]): #check if this word has a number in it. https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
            number_to_month[j] = -1
            text[j] = "".join([char for char in text[j] if char.isdigit()])
        for i in range(len(months)):
            if months[i] in text[j].lower():
                month_idx.append(i)
                month_to_number[j] = -1
                
    #check #1b: check for at least 2 months, and at least 2 numbers
    if (len(month_to_number) < 2) and (len(number_to_month) < 2):
        return None
    
    #forward pass: for each month, assume its number of seizures directly follows in the text (i.e. month at idx 6, number at idx 7)
    #for each month
    for month in month_to_number:
        #check if a number occurs at the next index and if that number has not already been taken by another month
        if (month+1) in number_to_month:
            if number_to_month[month+1] == -1:
                
                #check if this number is a year (4 digits > 1970)
                if float(text[month+1]) >= 1970:
                    number_to_month[month+1] = -2
                    
                    #if it is a year, then look at month+2.
                    if (month+2) in number_to_month:
                        if number_to_month[month+2] == -1:
                            month_to_number[month] = month+2
                            number_to_month[month+2] = month
                            
                #if not, assign the month to the number and vice versa
                else:
                    month_to_number[month] = month+1
                    number_to_month[month+1] = month

    #second pass: for each month, if it hasn't been assigned a number, look up to 3 indices backwards for a number.
    #if any of those 3 indices is a number and already taken by a different month, break for this month.
    for month in month_to_number:
        #ignore months that have been assigned a number
        if month_to_number[month] != -1:
            continue
        #look 3 indices backwards for numbers
        for i in range(month-1, month-4, -1):
            #prevent going below index 0
            if i < 0:
                break
            #if it is a number
            if i in number_to_month:
                #check if this number is taken. 
                #If so, break, because you likely would not have a sentence like "6, 7 in May" 
                #if the 7 was already taken by a different month, but 6 was left open
                #if the number is open, then assign it to the month
                if number_to_month != -1:
                    break
                else:
                    number_to_month[i] = month
                    month_to_number = i
                    break

    #check #2: check that at least two months have an associated number
    months_with_associations = []
    total_num_seizures = 0
    for month in month_to_number:
        if month_to_number[month] != -1: #if this month has an associated number
            #note the month, and accumulate the number of seizures
            months_with_associations.append(month)
            total_num_seizures += float(text[month_to_number[month]])
    if len(months_with_associations) < 2:
        return None
    
    print(f"Seizure Calendar detected: '{unprocessed_text}'. Summarization: {total_num_seizures} per {len(months_with_associations)} months")
    return total_num_seizures / len(months_with_associations)


def translate_summaries(predictions_and_summaries):
    """
    Converts the summaries generated by the T5 model into numbers or dates. Modifies, in place, the input pd.DataFrame
    Input:
        predictions_and_summaries: a pd.DataFrame with columns:
            prediction: the finetuned RoBERTa predictions to extract seizure frequency and date of last occurrence from note text
            summarization: the summarization of the prediction provided by the T5 model
            id: the (question/answer) id of the prediction and summarization
            'sz_per_month': the column to store the seizure frequency translations, as a float in units of per month
            'last_occurrence': the column to store the last seizure occurrence translation, as a datetime.            
    """
    for idx in predictions_and_summaries.index:
        try:
            #remove commas inside the summary (commas mess up formatting)
            predictions_and_summaries.loc[idx, 'summarization'] = predictions_and_summaries.loc[idx, 'summarization'].translate(str.maketrans('', '', ','))
            
            #check if it is a pqf or elo summary
            if 'pqf' in predictions_and_summaries.loc[idx, 'id'].lower():
                #if the summary is blank, skip it
                if predictions_and_summaries.loc[idx, 'summarization'] == "":
                    continue

                #check if it is a seizure calendar
                szCal = process_calendar(predictions_and_summaries.loc[idx, 'prediction'])
                if szCal is not None:
                    predictions_and_summaries.loc[idx,  'sz_per_month'] = szCal

                #if it isn't a seizure calendar, then process it normally
                else: 
                    
                    #split the summary by spaces
                    summary = predictions_and_summaries.loc[idx, 'summarization'].split()

                    #check if the 0th and 2nd indices are numeric
                    if not summary[0].isnumeric(): #if it isn't a number, it's probably a range. Split it by a dash
                        summary[0] = summary[0].split('-')[-1]
                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[0].isnumeric(): 
                            summary[0] = summary[0].split('/')[-1]
                    if not summary[2].isnumeric():
                        summary[2] = summary[2].split('-')[-1]
                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[2].isnumeric(): 
                            summary[2] = summary[0].split('/')[-1]

                    #if it is a "per visit" frequency, run the per-visit translator
                    if 'visit' in summary[-1].lower():
                        predictions_and_summaries.loc[idx, 'sz_per_month'] = 'calc_since_last_visit_pqf()'

                    #if it is a lifetime frequency, run the lifetime translator
                    elif 'life' in summary[-1].lower():
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = 'calc_lifetime_pqf()'

                    #we want to convert everything to per month.
                    #there are 12 months/year, and therefore 0.0833 years/month
                    #there are 365 days/year, 12 months/year, and therefore 30.4167 days/month
                    #there are 365 days/year, 12 months/year, 7 days/week, and therefore 4.3452 weeks/month
                    elif 'month' in summary[-1].lower(): #if it's per month, just calculate dx/dt using the existing values
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) / float(summary[2]) 
                    elif 'day' in summary[-1].lower() or 'night' in summary[-1].lower(): #if it's per day or per night, multiple the numerator by 30.4167
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 30.4167 / float(summary[2])
                    elif 'week' in summary[-1].lower(): #it it's per week, multiply the numerator by 4.3452
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 4.3452 / float(summary[2])
                    elif 'year' in summary[-1].lower(): #if it's per year, multiply the numerator by 0.0833
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 0.0833 / float(summary[2])
                    elif 'hour' in summary[-1].lower(): #if it's per hour, multiply the numerator by 24 and then by 30.4167
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 24 * 30.4167 / float(summary[2])
                    else:
                        print(f"ERROR - PQF timeframe unidentifiable. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
            
            #otherwise, check if it is a date of last seizure
            elif 'elo' in predictions_and_summaries.loc[idx, 'id'].lower():
                #if the summary is blank, skip it
                if predictions_and_summaries.loc[idx, 'summarization'] == "":
                    continue

                #split the summary by spaces
                summary = predictions_and_summaries.loc[idx, 'summarization'].split()

                #first, check if it is an "ago" last occurrence of the form X Y ago
                if 'ago' in summary[-1].lower():

                    #check if the 0th index is numeric
                    if not summary[0].isnumeric(): #if it isn't a number, it's probably a range. Split it by a dash and take the max
                        summary[0] = summary[0].split('-')[-1]

                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[0].isnumeric(): 
                            summary[0] = summary[0].split('/')[-1]

                    #get the Y (timeframe) of the ago "months, days, years, etc..."
                    if 'day' in summary[1] or 'night' in summary[1]: #if it is days ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0]))
                    elif 'week' in summary[1]: #if it is weeks ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*7)
                    elif 'month' in summary[1]: #if it is months ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*30.4167)
                    elif 'year' in summary[1]: #if it is years ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*365)
                    else:
                        print(f"ERROR - ELO 'ago' timeframe unidentifiable. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                    continue

                #otherwise, it must be some sort of date.
                days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
                is_month = False

                #the number of splits determines the format of the date
                if len(summary) == 1: #if the summary was only a single word,
                    #check if it is a number, in which case it must be a year
                    if summary[0].isnumeric():
                        #check if the year has 4 digits
                        if len(summary[0]) != 4:
                            print(f"ERROR - ELO summary had 1 item that suggests a year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                        #if it looks ok, convert it to a datetime. Python defaults the date to Jan 1st!!!!
                        summary = datetime.strptime(summary[0], '%Y')

                        #check if the year is within a reasonable range [1850-today]
                        if summary > datetime.today() or summary < datetime(year=1850, month=1, day=1):
                            print(f"ERROR - ELO summary had 1 item that suggests a year, but it was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                        #the year is then the last occurrence.
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = summary

                    #if it isn't a number, then it is a single month or single day
                    else:
                        try: #try to convert it as a month
                            elo_date = datetime.strptime(summary[0], '%B').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                            #if this proposed date is > the visit date, then move the year back by 1
                            if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)

                            predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                        except: #if you can't convert it as a month, try as a day
                            for i in range(len(days)): #check days
                                if days[i] in summary[0].lower():                         
                                    #count backwards from the visit date by day until we get to the right weekday
                                    visit_date = predictions_and_summaries.loc[idx,'visit_date']
                                    break_ct = 0
                                    while i != visit_date.weekday():
                                        visit_date -= timedelta(days=1)
                                        break_ct += 1
                                        if break_ct > 7:
                                            print(f"ERROR - date backtracking did not terminate in time. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                            break

                                    #this final date is the last occurrence
                                    predictions_and_summaries.loc[idx,  'last_occurrence'] = visit_date

                                    break

                        #if it was neither a or day, warn the user
                        if predictions_and_summaries.loc[idx,  'last_occurrence'] == -1:
                            print(f"ERROR - ELO summary had 1 item, but could translate neither a month, day, nor year. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                #if the date is given as 2 items it is either month, year; or month, day (year implicit);
                elif len(summary) == 2:

                    #check if the last item could be a year
                    if summary[-1].isnumeric() and len(summary[-1]) == 4:
                        try:
                            #if it is a year, then the first item must be a month. Otherwise, it doesn't make sense to give day, year without the month
                            if len(summary[0]) > 3:
                                elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %Y')
                            else:
                                elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%b %Y')

                            #if this proposed date is > the visit date, report an error
                            if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                print(f"ERROR - ELO summary had 2 items that suggest month year, but exceeds visit date. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                continue

                            #this final date is the last occurrence
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                        except:
                            print(f"ERROR - ELO summary had 2 items that suggest month year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                    #if it is not a year, then it must be a day. check if so
                    elif summary[-1].isnumeric() and (len(summary[-1]) <= 2 and len(summary[-1]) > 0):
                        day = int(summary[-1])

                        #check if the day is an acceptible range
                        if day < 1 or day > 31:
                            print(f"ERROR - ELO summary had 2 items that suggest month day, but day was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue
                        #if it is
                        else:
                            try:
                                #convert to datetime month day
                                if len(summary[0]) > 3:
                                    elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d').replace(year=predictions_and_summaries.loc[idx,'visit_date'].year)
                                else:
                                    elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%b %d').replace(year=predictions_and_summaries.loc[idx,'visit_date'].year)

                                #if this proposed date is > the visit date, then it must be from last year
                                if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                    elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)

                                #this final date is the last occurrence
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                            except:
                                print(f"ERROR - ELO summary had 2 items that suggest month day, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                continue
                    #else, there must have been a formatting error
                    else:
                        print(f"ERROR - ELO summary had 2 items, but could not find a suitable timeframe pair. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                        continue

                #if the date is given as three items, it must be in for month, day, year
                elif len(summary) == 3:
                    try:
                        #check if it's an "X or Y" format. If so, take the last one
                        if summary[1].lower() == 'or':
                            if summary[-1].isnumeric():
                                #check if the year has 4 digits
                                if len(summary[-1]) != 4:
                                    print(f"ERROR - ELO summary had 1 item that suggests a year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                    continue

                                #if it looks ok, convert it to a datetime. Python defaults the date to Jan 1st!!!!
                                summary = datetime.strptime(summary[-1], '%Y')

                                #check if the year is within a reasonable range [1850-today]
                                if summary > datetime.today() or summary < datetime(year=1850, month=1, day=1):
                                    print(f"ERROR - ELO summary had 1 item that suggests a year, but it was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                    continue

                                #the year is then the last occurrence.
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = summary
                            #if it isn't a number, then it is a single month or single day
                            else:
                                #try to convert it as a month
                                if len(summary[-1]) > 3:
                                    elo_date = datetime.strptime(summary[-1], '%B').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                                else:
                                    elo_date = datetime.strptime(summary[-1], '%b').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                                #if this proposed date is > the visit date, then move the year back by 1
                                if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                    elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date
                        elif len(summary[-1]) == 4:
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d %Y')
                        else:
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d %y')
                    except:
                        print(f"ERROR - ELO summary had full 3-part date, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                        continue
                else:
                    print(f"ERROR - ELO summary had more than 3 parts. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                    continue

                #if the ELO could not be determined, warn the user
                if predictions_and_summaries.loc[idx,  'last_occurrence'] == -1:
                    print(f"ERROR - ELO could not be translated. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")

            else:
                print(f"ERROR - unidentifiable id. id: predictions_and_summaries.loc[idx, 'id']. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
        except Exception as e:
            print(f"ERROR - some problem occurred, skipping. Filename: {predictions_and_summaries.loc[idx, 'filename']}. Summary: {predictions_and_summaries.loc[idx, 'summarization']}\nException: {e}")
            continue
            
#go through and process since last visits
#sort predictions_and_summaries by pat_id and then by visit_date
def process_since_last_visit(predictions_and_summaries):
    """
    Translates "since last visit" frequencies into floats of units per month by attempting to identify a patient's last visit
    Input:
        predictions_and_summaries: pd.DataFrame that was just passed into translate_summaries()
    Returns:
        pd.DataFrame
    """
    
    predictions_and_summaries = predictions_and_summaries.sort_values(['pat_id', 'visit_date'])
    for idx in range(len(predictions_and_summaries)):
        try:
            #skip non since-last-visit pqfs
            if predictions_and_summaries.iloc[idx]['sz_per_month'] != 'calc_since_last_visit_pqf()':
                continue

            #if the summary is a since last visit, get the date of the last visit. Because rows are sorted by visit date, it should be the row directly above
            #first, check that the (above) index is within bounds
            if idx - 1 < 0:
                print(f"Error: previous index out of bounds. idx: {idx}")
                predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = -2
                continue
            #check if the row above is of the right patient
            if not (predictions_and_summaries.iloc[idx-1]['pat_id'] == predictions_and_summaries.iloc[idx]['pat_id']):
                print(f"Error: previous index is a different patient. idx: {idx}")
                predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = -2
                continue
            #get the previous visit date and the time passed since then (in months)
            time_passed = (predictions_and_summaries.iloc[idx]['visit_date'] - predictions_and_summaries.iloc[idx-1]['visit_date']).days / 30.4167

            if time_passed == 0:
                print(f"Error: no time difference between current and last visit. idx: {idx}")
                continue

            #calculate the frequency dx/dt
            summary = predictions_and_summaries.iloc[idx]['summarization'].split()
            #if it isn't a number, it's probably a range. Split it by a dash
            if not summary[0].isnumeric(): 
                summary[0] = summary[0].split('-')[-1]

                #if it still isn't a number, it could be an "or", in which case split it by '/'
                if not summary[0].isnumeric(): 
                    summary[0] = summary[0].split('/')[-1]

            predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = float(summary[0]) / time_passed
        except:
            print(f"ERROR COULD NOT PROCESS THIS SINCE_LAST_VISIT: {predictions_and_summaries.iloc[idx]['summarization']}")
            continue
    return predictions_and_summaries

#loads a paragraph of text from a medical note
def get_paragraph_with_max_token_length(whitelist_regex, blacklist_regex, note_text, note_author, pat_id, visit_date, note_id, model_path, splitter="  ", max_token_length=512, debug=False):
    """
    Load a paragraph of text given a medical note. Truncates at maximum token length
    Input:
        whitelist_regex: What keywords do we want to split the note on (e.g. "HPI") and include as a paragraph
        blacklist_regex: Waht keywords do we want to split the note on and exclude as a paragraph
        note_text: The note's text
        note_author: The note's author
        pat_id: The patient's ID
        visit_date: The date of the visit
        note_id: The note's identifier
        model_path: The path to the transformer model 
        splitter: How are new lines in the note text split from each other?
        max_token_length: How many tokens to keep for truncation
    Returns: a dictionary containing the document's info
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    #split the document into lines
    sentences = note_text.strip('"').split(splitter)
    
    #skip notes that are less than or equal to 5 lines long
    if len(sentences) <= 5:
        return None
        
    #Dictionary to store relevant information of the document
    document = {}
    document['filename'] = f"{pat_id}_{note_id}_{note_author}_{visit_date[:10]}"
    document['note_author'] = note_author
    
    #scan through each line and find indices where it contains a desired header
    whitelist_indices = []
    blacklist_indices = []
    header_indices = []
    for i in range(len(sentences)):
        substr = sentences[i].strip()[:30]
        if re.search(whitelist_regex, substr):
            whitelist_indices.append(i)
            header_indices.append(i)
        elif re.search(blacklist_regex, substr):
            blacklist_indices.append(i)
            header_indices.append(i)
    header_indices.append(-1)
    
    #extract the paragraphs starting from a whitelist header until the next white or blacklisted header. 
    extract_counter = 0    
    for i in range(1, len(header_indices)):
        if header_indices[i-1] in blacklist_indices:
            continue
        elif header_indices[i] == -1:
            #get the paragraph text
            doc_text = " ".join([line.strip() for line in sentences[header_indices[i-1]:] if line != ""])
            
            #truncate to max_token_length
            doc_text = tokenizer.decode(tokenizer(doc_text, max_length=512, stride=128, truncation='do_not_truncate', add_special_tokens=False)['input_ids'][:max_token_length])
            
            #if there are at least 250 characters in the resultant text, keep it in consideration
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
        else:
            #get paragraph of text
            doc_text = " ".join([line.strip() for line in sentences[header_indices[i-1]:header_indices[i]] if line != ""])
            
            #truncate to max_token_length
            doc_text = tokenizer.decode(tokenizer(doc_text, max_length=512, stride=128, truncation='do_not_truncate', add_special_tokens=False)['input_ids'][:max_token_length])
            
            #if there are at least 250 characters in the resultant text, keep it in consideration
            if len(doc_text) > 250:
                document[extract_counter] = doc_text
                extract_counter += 1
    
    #if no paragraphs were found, then take the first max_token_length tokens of the text.
    if len(document) <= 2:
        doc_text = "".join([line.strip() + "\n" for line in sentences if line != ""])
        #truncate to max_token_length
        doc_text = tokenizer.decode(tokenizer(doc_text, max_length=512, stride=128, truncation='do_not_truncate', add_special_tokens=False)['input_ids'][:max_token_length])
        document[extract_counter] = doc_text
        extract_counter += 1
    
    if debug:
        return [document, sentences, whitelist_indices, blacklist_indices, header_indices]
    else:
        return document

    
def replace_implicits(text, implicit_converter):
    """
    Replaces words like "many" and "some" with the desired numerical conversion
    Input: 
        text: The text to process
        implicit_converter: The conversion dictionary
    Returns:
        The converted text
    """
    for imp in implicit_converter:
            text = text.replace(str(imp), str(implicit_converter[imp]))
    return text

    
def generate_aggregate_visit_table(all_pats, min_num_visits=5):
    """
    Creates a table of aggregate visits for a dictionary of seeds of patients
    Input:
        all_pats: A dictionary, with keys of seeds and values of lists of patients
        min_num_visits: The minimum number of visits needed to be included in the table
    Return:
        A table of aggregate visits, where the rows are patients and columns are visits
    """
    #sort each patient's visits by visit date
    for seed in all_pats:
        for pat_idx in range(len(all_pats[seed])):
            all_pats[seed][pat_idx].visits.sort(key=lambda x:x.visit_date)

    #construct the patient visit dataframe
    visit_tbl = {}
    #for each seed
    for seed in all_pats:
        visit_tbl[seed] = {}
        #for each patient get their visits
        for pat in all_pats[seed]:
            visit_tbl[seed][pat.pat_id] = pat.visits

        #convert to a dataframe
        visit_tbl[seed] = pd.DataFrame.from_dict(visit_tbl[seed], orient='index')

        #remove all patients with < 5 visits
        visit_tbl[seed] = visit_tbl[seed].loc[~pd.isnull(visit_tbl[seed][min_num_visits-1])]

    #conduct plurality voting across each seed to get final visit info
    #preallocate
    plurality_voting_tbl = pd.DataFrame().reindex_like(visit_tbl[2])
    #create Aggregate_Visit objects for each visit across the seeds
    for pat_id in visit_tbl[2].index:
        for visit_num in visit_tbl[2].columns:
            if not pd.isnull(visit_tbl[2].loc[pat_id, visit_num]):        
                plurality_voting_tbl.loc[pat_id, visit_num] = Aggregate_Visit(pat_id=pat_id, 
                                                                             note_id=visit_tbl[2].loc[pat_id, visit_num].note_id,
                                                                             visit_date=visit_tbl[2].loc[pat_id, visit_num].visit_date,
                                                                             author=visit_tbl[2].loc[pat_id, visit_num].author, 
                                                                             all_visits=[visit_tbl[seed].loc[pat_id, visit_num] for seed in visit_tbl])

    #sort our (aggregate) visits by visit_date
    return plurality_voting_tbl.sort_values(by=0, axis=0, key=lambda x: [agg.visit_date for agg in x])


def aggregate_patients_and_visits(all_pats):
    """Aggregates patients and visits from dictionary of array of patients all_pats, where each key is a different seed"""

    #initialize the array of Aggregate_Patients
    agg_pats = []
    
    #for simplicity, get the first key
    k = list(all_pats.keys())[0]
    
    #create Aggregate_Patients and fill in their Aggregate_Visits
    for i in range(len(all_pats[k])):
        new_Agg_Pat = Aggregate_Patient(all_pats[k][i].pat_id)
        
        #get aggregate visits
        for j in range(len(all_pats[k][i].visits)):
            new_Agg_visit = Aggregate_Visit(aggregate_patient=new_Agg_Pat,
                                            note_id = all_pats[k][i].visits[j].note_id,
                                            visit_date = all_pats[k][i].visits[j].visit_date,
                                            author = all_pats[k][i].visits[j].author,
                                            all_visits = [all_pats[seed][i].visits[j] for seed in all_pats.keys()]
                                           )
        
        agg_pats.append(new_Agg_Pat)
            
    return agg_pats


def generate_state_names(num_states, order):
    #generate the names of the states (one-hot-encoded) of our markov-like process (i.e. 00, 01, 10, 11, etc...)
    state_names = []
    for i in range(np.power(num_states, order)):
        state_names.append(np.base_repr(i, num_states).rjust(order, '0'))
    return state_names

def generate_markov_adjacency_matrix(num_states, markov_order, transition_counts):
    """
    Generate our 2D adjacency matrix for our markov-like chain
        row/col indices will sequentially list out the markov states. 
        i.e. for 2nd order, P(state(0,0) -> state(0,1)) = C(1|0,0) = transition_counts[1,0,0]
    Input:
        num_states: the number of states
        markov_order: the order of our chain
        transition_counts: counts of the transitions in the chain
    """
    
    assert markov_order > 1, "This function requires a markov chain of at least 2nd order"

    #generate the state names
    state_names = generate_state_names(num_states, markov_order)
    
    #raw matrix to create network graphs
    raw_adjacency_mat = pd.DataFrame(0, index=state_names, columns=state_names)
    #simplified matrix to show only transitions to the class value
    simplified_adjacency_mat = pd.DataFrame(0, index=state_names, columns=list(range(num_states)))
    
    #for each source node -> destination node transition, get its count
    for source in raw_adjacency_mat.index:
        for dest in raw_adjacency_mat.columns:
            if source[1:] == dest[:-1]:
                raw_adjacency_mat.loc[source, dest] = transition_counts[tuple([int(idx) for idx in source+dest[-1]])]
                simplified_adjacency_mat.loc[source, int(dest[-1])] = transition_counts[tuple([int(idx) for idx in source+dest[-1]])]

    #normalize each source->dest row to get probabilities
    raw_adjacency_mat = raw_adjacency_mat.div(raw_adjacency_mat.sum(axis=1), axis=0)
    simplified_adjacency_mat = simplified_adjacency_mat.div(simplified_adjacency_mat.sum(axis=1), axis=0)  
    
    #replace nan with 0
    raw_adjacency_mat = raw_adjacency_mat.fillna(0)
    simplified_adjacency_mat = simplified_adjacency_mat.fillna(0).sort_values(by=0)
    
    return raw_adjacency_mat, simplified_adjacency_mat


def generate_visit_markovian_table(agg_pats, order=3, states=set([0,1]), seq_to_sym={'0':r"$\bigcirc$", '1':r"$\blacksquare$", '2':'IDK'}, save_path = None, no_plot = False, sort_output = True):
    """
    Generates a markov-like table with the probability of seizure freedom given the previous number of visits
    Input:
        agg_pats: The aggregate patients
        order: The order of the chain
        states: The states we care about
        seq_to_sym: Visualize states as symbols for convenience
        save_path: Where to save the final plot
        no_plot: Do we want to plot?
        sort_output: Sort the final table
    """
    
    #markov chain parameters
    order = 3
    num_states = len(states)
    markov_ct = np.zeros([num_states for i in range(order)]+[num_states])
    num_pats_used = 0
    
    #for each patient, create their visit timeline
    for pat in agg_pats:    
        #these lists hold the date, classification, and number of ASMs at each visit
        visit_dates = []
        visit_classifications = []
        
        #was this patient ultimately used in the chain?
        pat_used = False

        #Iterate through the visits and get the date and classifications
        for visit in pat.aggregate_visits:
            visit_dates.append(visit.visit_date)
            visit_classifications.append(visit.hasSz)

        #sort the visits in order of time
        sort_indices = np.argsort(visit_dates)
        visit_dates = np.array(visit_dates)[sort_indices]
        visit_classifications = np.array(visit_classifications)[sort_indices]    

        #begin counting for our markov chain
        for i in range(len(visit_dates) - (order+1)):
            #ignore all cases where a state is not in the states list
            state_compare = set(visit_classifications[i:i+(order+1)]) - states

            #increment the markov counter
            if (state_compare in states) or (len(state_compare) == 0):           
                markov_ct[tuple(visit_classifications[i:i+order+1])] += 1
                pat_used = True
                
        num_pats_used += int(pat_used)

    #calculate probability of seizures given previous order months of seizure freedom, etc...
    prob_sz = np.zeros(markov_ct.shape) #0 = szFree, 1 = hasSz.
    sum_sz = np.empty(markov_ct.shape, dtype=object)
    ct_flat = markov_ct.flatten()
    for i in range(len(ct_flat)):
        ct_idx = np.unravel_index(i, markov_ct.shape)
        prob_sz[ct_idx] = ct_flat[i] / np.sum(markov_ct[ct_idx[:-1]])
        sum_sz[ct_idx] = f"{int((prob_sz[ct_idx]*100).round())}%\n({int(np.sum(markov_ct[ct_idx[:-1]]))})"

    #convert the probability matrix to a dataframe
    prob_sz_df = {}
    heatmap_labels = {}
    prob_flat = prob_sz.flatten()
    sum_flat = sum_sz.flatten()
    for i in range(len(prob_sz.flatten())):
        idx = np.unravel_index(i, prob_sz.shape)

        #create a sequence of symbols if this sequence has already been stored
        sequence = r"".join(seq_to_sym[ch] for ch in str(idx[:-1]) if ch.isdigit())

        #the sum_flat label has both the probability value and the count of examples
        labels = sum_flat[i].split('\n')

        #append to the sequence the count of examples and format for symbols
        sequence += f"\n{labels[1]}"
        sequence = f"{sequence}"

        #check if this sequence has been used already
        if str(sequence) not in prob_sz_df:
            prob_sz_df[sequence] = {}
            heatmap_labels[sequence] = {}

        #insert the value into the row/column of the dataframe (once created)    
        prob_sz_df[sequence][seq_to_sym[f'{idx[-1]}']] = prob_flat[i]
        heatmap_labels[sequence][seq_to_sym[f'{idx[-1]}']] = labels[0]

    #finish converting to dataframe
    prob_sz_df = pd.DataFrame(prob_sz_df).transpose()
    heatmap_labels = pd.DataFrame(heatmap_labels).transpose()

    #sort dataframe columns
    if sort_output:
        prob_sz_df = prob_sz_df[np.sort(prob_sz_df.columns)].sort_values(by=seq_to_sym['0'])
        heatmap_labels = heatmap_labels[np.sort(heatmap_labels.columns)].loc[prob_sz_df.index]

    #plot
    if not no_plot:
        fig = plt.figure(figsize = (6,order*5))
        ax = sns.heatmap(prob_sz_df, annot=heatmap_labels, vmin=0, vmax=1, linewidth=0.25, linecolor='#303030', fmt="",
                         cbar_kws={'label': 'Probability'}, cmap=sns.color_palette("Blues", as_cmap=True))
        ax.xaxis.tick_top()
        ax.set_xlabel(f"Next Visit Classification\n{seq_to_sym['0']} = Seizure Free\n{seq_to_sym['1']} = Has Seizures\n")
        ax.xaxis.set_label_position('top') 
        plt.ylabel('(Ordered) Previous Visit Classifications')
        plt.yticks(rotation=0)
        plt.title(f"Probability of Having Seizures or Being Seizure Free\nGiven a Patient's Previous {order} Visits\n")

        if save_path:
            plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
            
        print(f"Number of patients taken into account in the Markov Chain: {num_pats_used}")
        
    return prob_sz_df


#create a dataframe with indices from 0 up to the maximum days after the breakpoint
def generate_survival_table(all_final_to_breakpoint_time_diff, all_future_visit_dates, all_future_visit_classifications, all_breakpoint_visit_dates, days_cutoff = None, default_value=0):
    """
    Calculates the survival table for the kaplan-meier time to event analysis
    """
    
    #find which patients meet the inclusion criteria
    pat_indices_used = [i for i in range(len(all_future_visit_dates)) if (all_future_visit_dates[i][-1] - all_breakpoint_visit_dates[i] >= days_cutoff)]
    
    #rows = patients, columns = days. We need to +1, because the range must be inclusive on both size, [0, max_days]
    survival_table = np.ones((len(pat_indices_used), np.max(all_final_to_breakpoint_time_diff).days+1)) * default_value

    #for each patient, add their influence to the survival table
    pat_ct = 0
    for i in pat_indices_used:

        #for each visit of each patient, accumulate
        for j in range(len(all_future_visit_dates[i])):
            
            #calculate how much time has passed since the breakpoint
            days_after = (all_future_visit_dates[i][j] - all_breakpoint_visit_dates[i]).days

            #add in the patient's visit classification. We assume everyone is szFree until their visit.
            survival_table[pat_ct, days_after] = all_future_visit_classifications[i][j]
            
            #if the visit was hasSz, then the patient leaves the survival curve - we stop processing this patient
            if all_future_visit_classifications[i][j] != default_value:
                break
                
        #if the patient is out of visits, or they were hasSz, then set all future values to NaN
        survival_table[pat_ct, days_after+1:] = np.nan
        pat_ct += 1
        
    #contract the survival table -> all 1's are where people became hasSz, and all 0's are where they stayed seizure free
    hasSz_transition = np.nansum((survival_table!=default_value) & ~(np.isnan(survival_table)), axis=0) #people who became hasSz that day
    szFree_population = np.nansum(survival_table==default_value, axis=0) + hasSz_transition #people who stayed szFree that day + people who changed = total people who started the day szFree
        
    #calculate the survival table
    szFree_survival = 1 - hasSz_transition/szFree_population
    szFree_probability = []
    for i in range(len(szFree_survival)):
        if i == 0:
            szFree_probability.append(szFree_survival[0])
        else:
            szFree_probability.append(szFree_survival[i] * szFree_probability[-1])

    return pd.DataFrame({'days_after':np.arange(0, np.max(all_final_to_breakpoint_time_diff).days+1), 'szFree':szFree_probability, 'num_pats':szFree_population}).set_index('days_after'), pat_indices_used, survival_table
    

def generate_markovian_pat_history_survival_curve(agg_pats, history_months=6, survival_cutoff_years=5, plot_xlim_years=5, save_path = None, no_plot = False):
    """
    Generates the kaplan-meier survival curve
    Input:
        agg_pats: the Aggregate_Patients
        history_months: How many months of history do we want
        survival_cutoff_years: Inclusion criteria - how many years of data after the breakpoint do patients need to be included?
        plot_xlim_years: When to cutoff the survival curve
        save_path: Where to save the files
        no_plot: Stop plotting
    """
    markov_ct = np.zeros((2,2))
    all_breakpoint_visit_dates = []
    all_future_visit_dates = []
    all_future_visit_classifications = []
    all_final_to_breakpoint_time_diff = []
    all_six_mo_summaries = []
    all_classifications_across_prev_visits = []
    

    #for each patient, count the number of times they were seizure free or having seizures in a roughly history_months month interval
    for pat in agg_pats:

        #a patient needs to have at least 3 visits to be considered
        if len(pat.aggregate_visits) < 3:
            continue

        #sort the visits by visit date
        aggregate_visits_sorted = sorted(pat.aggregate_visits, key=lambda x:x.visit_date)

        #iterate through their visits - if there is a roughly history_months month gap between their current and previous visit, then it's a useable datapoint
        #or, if there is a roughly history_months month gap across multiple visits, that is also a useable datapoint
        #we will accept +/- half a month
        i = 0
        while i  < len(aggregate_visits_sorted) - 1:

            #internal loop counter
            j=i

            #accumulate time between visits for this window
            interval_time = timedelta(days=0)
            #accumulate the classification between visits
            classifications_across_prev_visits = []

            #while we're still under the history_months month threshold, and we haven't exceeded our visit array
            while interval_time < timedelta(days=history_months*30.4167 - 30.4167/2) and j < len(aggregate_visits_sorted) - 1:

                #accumulate the amount of time between visits
                interval_time += aggregate_visits_sorted[j+1].visit_date - aggregate_visits_sorted[j].visit_date

                #get the next visit classification
                classifications_across_prev_visits.append(aggregate_visits_sorted[j+1].hasSz)

                #go into the future
                j += 1

            #if any of the visits were unclassified,
            #skip
            if (2 in classifications_across_prev_visits):
                i+=1
                continue
            else:
                #skip patients who have no more data
                if j >= len(aggregate_visits_sorted) - 1:
                    break

                #get the reminaing visits if they exist
                future_visit_classifications = [vis.hasSz for vis in aggregate_visits_sorted[j+1:]]   

                #skip patients who have idk classifications in their future
                if 2 in future_visit_classifications:
                    i+=1
                    continue

                all_future_visit_classifications.append(future_visit_classifications)
                all_future_visit_dates.append([vis.visit_date for vis in aggregate_visits_sorted[j+1:]])

                #what is the time at the breakpoint from history_months months in the past to the future?
                all_breakpoint_visit_dates.append(aggregate_visits_sorted[j].visit_date)
                final_to_breakpoint_time_diff = aggregate_visits_sorted[-1].visit_date - aggregate_visits_sorted[j].visit_date
                all_final_to_breakpoint_time_diff.append(final_to_breakpoint_time_diff)

                #get the patient's status from the last six months
                six_mo_summary = int((1 in classifications_across_prev_visits))
                all_classifications_across_prev_visits.append(classifications_across_prev_visits)
                all_six_mo_summaries.append(six_mo_summary)

                #accumulate the counter
                markov_ct[six_mo_summary, int((1 in future_visit_classifications))] += 1 * final_to_breakpoint_time_diff.days

                break
            
    #calculate probability of seizures given previous order months of seizure freedom, etc...
    prob_sz = np.zeros(markov_ct.shape) #0 = szFree, 1 = hasSz.
    sum_sz = np.empty(markov_ct.shape, dtype=object)
    ct_flat = markov_ct.flatten()
    for i in range(len(ct_flat)):
        ct_idx = np.unravel_index(i, markov_ct.shape)
        prob_sz[ct_idx] = ct_flat[i] / np.sum(markov_ct[ct_idx[:-1]])
        sum_sz[ct_idx] = f"{prob_sz[ct_idx].round(decimals=3)}\n({np.sum(markov_ct[ct_idx[:-1]])})"

    #convert the probability matrix to a dataframe
    prob_sz_df = {}
    heatmap_labels = {}
    prob_flat = prob_sz.flatten()
    sum_flat = sum_sz.flatten()
    for i in range(len(prob_sz.flatten())):
        idx = np.unravel_index(i, prob_sz.shape)

        #check if this sequence has already been stored
        sequence = "".join(ch for ch in str(idx[:-1]) if ch.isdigit())
        if str(sequence) not in prob_sz_df:
            prob_sz_df[sequence] = {}
            heatmap_labels[sequence] = {}

        #insert the value into the row/column of the dataframe (once created)    
        prob_sz_df[sequence][idx[-1]] = prob_flat[i]
        heatmap_labels[sequence][idx[-1]] = sum_flat[i]

    #finish converting to dataframe
    prob_sz_df = pd.DataFrame(prob_sz_df).transpose()
    heatmap_labels = pd.DataFrame(heatmap_labels).transpose()

    #plot
    if not no_plot:
        fig = plt.figure(figsize = (6,5))
        ax = sns.heatmap(prob_sz_df, annot=heatmap_labels, vmin=0, vmax=1, linewidth=0.25, linecolor='#303030', fmt="",
                         cbar_kws={'label': 'Probability'}, cmap=sns.color_palette("Blues", as_cmap=True))
        ax.xaxis.tick_top()
        ax.set_xlabel('Future Classification\n0 = Seizure Free\n1 = Has Seizures\n')    
        ax.xaxis.set_label_position('top') 
        plt.ylabel(f'Previous {history_months} Months Classification')
        plt.yticks(rotation=0)
        plt.title(f"Probability of Having Seizures or Being Seizure Free\nGiven a Patient's Previous {history_months} Months\nWeighted By Time Afterwards\n")
        plt.show()

    #convert to numpy array for indexing
    all_final_to_breakpoint_time_diff = np.array(all_final_to_breakpoint_time_diff)
    all_future_visit_dates = np.array(all_future_visit_dates, dtype='object')
    all_future_visit_classifications = np.array(all_future_visit_classifications, dtype='object')
    all_classifications_across_prev_visits = np.array(all_classifications_across_prev_visits, dtype='object')
    all_breakpoint_visit_dates = np.array(all_breakpoint_visit_dates)
    all_six_mo_summaries = np.array(all_six_mo_summaries)

    print("\n\n\n")
    
    #calculate the survival tables
    cutoff = timedelta(days=365*survival_cutoff_years) #survival_cutoff_years years of data minimum
    hasSz_start = all_six_mo_summaries == 1
    hasSz_start_survival, hasSz_indices_used, hasSz_survival = generate_survival_table(all_final_to_breakpoint_time_diff[hasSz_start], all_future_visit_dates[hasSz_start], all_future_visit_classifications[hasSz_start], all_breakpoint_visit_dates[hasSz_start], cutoff, 1)
    
    szFree_start = all_six_mo_summaries == 0
    szFree_start_survival, szFree_indices_used, szFree_survival = generate_survival_table(all_final_to_breakpoint_time_diff[szFree_start], all_future_visit_dates[szFree_start], all_future_visit_classifications[szFree_start], all_breakpoint_visit_dates[szFree_start], cutoff, 0)

    print(f"Final probability szFree -> breakthrough seizures after {plot_xlim_years} years: {1-(szFree_start_survival.loc[plot_xlim_years*365])}")
    print(f"Final proportion hasSz => SzFree visit after {plot_xlim_years} years: {1 - (hasSz_start_survival.loc[plot_xlim_years*365])}")

    hasSz_start_survival.index/=365 #convert from days after to years after
    szFree_start_survival.index/=365
    hasSz_start_survival.index.names=['years_after']
    szFree_start_survival.index.names=['years_after']

    
    #panel a and b
    #probability of sustained seizure freedom
        #probability of a breakthrough seizure
    #probability of non seizure freedom?
        #probability of a seizure free visit

    #create a survival plot
    if not no_plot:
        fig=plt.figure(figsize = (6,6))
        plt.title(f"Seizure Free Visits for Patients with\n{history_months} Months of Prior History And at least {survival_cutoff_years} Years of Data Afterwards.")
        plt.plot(1-hasSz_start_survival['szFree'], c='#d95f02')
        plt.xlabel(f"Years After {history_months} Months of History")
        plt.ylabel("Probability of a Seizure Free Visit")
        plt.legend([f"n={hasSz_start_survival.loc[0, 'num_pats']}"])
        plt.xlim([0, plot_xlim_years])
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(f"{save_path}_szfree_vis.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{save_path}_szfree_vis.pdf", dpi=600, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
            
        fig=plt.figure(figsize = (6,6))
        plt.title(f"Breakthrough Seizures For Patients with\n{history_months} Months of Prior History And at least {survival_cutoff_years} Years of Data Afterwards.")
        plt.plot(1-szFree_start_survival['szFree'], c='#1b9e77')
        plt.xlabel(f"Years After {history_months} Months of History")
        plt.ylabel("Probability of a Breakthrough Seizure")
        plt.legend([f"n={szFree_start_survival.loc[0, 'num_pats']}"])
        plt.xlim([0, plot_xlim_years])
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(f"{save_path}_breakthrough.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{save_path}_breakthrough.pdf", dpi=600, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
        
    return [prob_sz_df, hasSz_start_survival, szFree_start_survival, all_classifications_across_prev_visits[hasSz_indices_used], all_classifications_across_prev_visits[szFree_indices_used], all_future_visit_classifications[hasSz_indices_used], all_future_visit_classifications[szFree_indices_used], hasSz_survival, szFree_survival]