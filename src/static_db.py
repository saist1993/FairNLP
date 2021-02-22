"""
A file which stores attribute useful for WEAT. In later run this will also hold static information required for other
form of analysis
"""
# The same function also exists in util. Had to copy because of circular imports.
def combine_two_dictionaries(dict1, dict2):
    temp = {}
    for key in set().union(dict1, dict2):
        if key in dict1: temp.setdefault(key, []).extend(dict1[key])
        if key in dict2: temp.setdefault(key, []).extend(dict2[key])

    # Create final dictionary without repeating items in value.
    new_dict = {}
    for key, value in temp.items():
        new_dict[key] = list(set(value))

    return new_dict

def clean(input_str):
    return [i.strip () for i in input_str.split(",")]

# The sets of words concerning a particular attribute class are taken from Popovic, Lemmerich and Strohmaier (2020).
dict1 = {'pleasant':
             ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
                      'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle',
                      'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation', 'joy', 'wonderful'],
         'unpleasant': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison',
                        'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty',
                        'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison', 'terrible', 'horrible'],
         'instruments': ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin',
                         'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano',
                         'viola', 'bongo', 'flute', 'horn', 'saxophone'],
         'weapons': ['arrow', 'club', 'gun', 'missile', 'spear', 'dagger', 'harpoon', 'pistol', 'sword', 'blade',
                     'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon',
                     'grenade', 'mace', 'slingshot', 'whip'],
         'career': ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
         'family': ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives'],
         'science': ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy'],
         'art': ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama'],
         'intellectual_words': ['resourceful', 'inquisitive', 'sagacious', 'inventive', 'adaptable', 'intuitive',
                                'analytical', 'imaginative', 'shrewd', 'thoughtful', 'smart', 'clever', 'brilliant',
                                'logical', 'intelligent', 'apt', 'genius', 'wise', 'stupid', 'dumb', 'dull', 'clumsy',
                                'foolish', 'naive', 'unintelligent'],
         'appearance_words': ['alluring', 'voluptuous', 'blushing', 'homely', 'plump', 'sensual', 'gorgeous', 'slim',
                              'bald', 'athletic', 'fashionable', 'stout', 'ugly', 'muscular', 'slender', 'feeble',
                              'handsome', 'healthy', 'attractive', 'fat', 'weak', 'thin', 'pretty', 'beautiful',
                              'strong'],
         'shy': ['soft', 'quiet', 'compromising', 'rational', 'calm', 'kind', 'agreeable', 'servile', 'pleasant',
                 'cautious', 'friendly', 'supportive', 'nice', 'mild', 'demure', 'passive', 'indifferent'],
         'aggressive': ['shrill', 'loud', 'argumentative', 'irrational', 'angry', 'abusive', 'obnoxious', 'controlling',
                        'nagging', 'brash', 'hostile', 'mean', 'harsh', 'sassy', 'aggressive', 'opinionated',
                        'domineering'],
         'competent': ['competent', 'productive', 'effective', 'ambitious', 'active', 'decisive', 'strong', 'tough'],
         'incompetent': ['incompetent', 'unproductive', 'ineffective', 'passive', 'indecisive', 'weak', 'gentle',
                         'timid'],
         'likeable': ['agreeable', 'fair', 'honest', 'trustworthy', 'selfless', 'accommodating'],
         'unlikeable': ['abrasive', 'conniving', 'manipulative', 'dishonest', 'selfish', 'pushy']}

# Follows from Caliskan et al 2017 (10.1126/science.aal4230).

dict2 = {
    'instruments': ["bagpipe", "cello", "guitar", "lute", "trombone", "banjo", "clarinet", "harmonica", "mandolin",
                    "trumpet", "bassoon", "drum", "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano",
                    "viola", "bongo", "flute", "horn", "saxophone", "violin"],
    'weapons': ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", "harpoon", "pistol", "sword", "blade",
                "dynamite", "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", "cannon",
                "grenade", "mace", "slingshot", "whip"],
    'flowers': ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose",
                "blue-bell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet",
                "carnation", "gladiola", "magnolia", "petunia", "zinnia"],
    'insects': ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula",
                "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp",
                "blackfly", "dragonfly", "horsefly", "roach", "weevil"],
    'pleasant': ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
                 "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise",
                 "family", "happy", "laughter", "paradise", "vacation"],
    'unpleasant': ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink",
                   "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly",
                   "cancer", "kill", "rotten", "vomit", "agony", "prison"],
    'career': ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
    'family': ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives'],
    'math': ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
    'arts': ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
    'male': ["male", "man", "boy", "brother", "he", "him", "his", "son"],
    'female': ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"],
    'science': ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"],
    'arts_2': ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"],
    'male_2': ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"],
    'female_2': ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"],
    'mental_disease': ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"],
    'physical_disease': ["sick", "illness", "influenza", "disease", "virus", "cancer"],
    'temporary': ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional"],
    'permanent': ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"],
}


intersectional_bias = {
    'african_american_females': clean("Aisha, Keisha, Lakisha, Latisha, Latoya, Malika, Nichelle,Shereen, Tamika, Tanisha, Yolanda, Yvette"),
    'african_american_males': clean("Alonzo, Alphonse, Hakim, Jamal, Jamel, Jerome, Leroy, Lionel,Marcellus, Terrence, Tyrone, Wardell"),
    'european_american_females': clean("Carrie,  Colleen,  Ellen,  Emily,  Heather,  Katie,  Megan,Melanie, Nancy, Rachel, Sarah, Stephanie"),
    'european_american_males': clean("Andrew, Brad, Frank, Geoffrey, Jack, Jonathan, Josh, Matthew,Neil, Peter, Roger, Stephen"),
    "african_american_females_ibd": clean("aggressive, assertive, athletic, big butt, confident, dark skinned, fried chicken, ghetto, loud, overweight, promiscuous, unfeminine, unintelligent, unrefined"),
    "african_american_males_ibd": clean("athletic, criminals, dangerous, dark skinned, gangsters, hypersexual, lazy, loud, poor, rapper, tall, unintelligent, violent"),
    "european_american_females_ibd": clean("arrogant, attractive, blond, ditsy,emotional, feminine, high status, intelligent, materialistic, petite, racist, rich, submissive,tall"),
    "european_american_males_ibd": clean("all American, arrogant, assertive,attractive, blond, educated, high status, intelligent, leader, privileged, racist, rich, sexist,successful, tall"),
    "african_american_females_eibd" : clean("dark skinned, hyper sexual,rapper"),
    "african_american_males_eibd": clean("aggressive, assertive, big butt, confident, dark skinned, fried-chicken, overweight, promiscuous, unfeminine"),
    "european_american_females_eibd": clean("ditsy"),
    "european_american_males_eibd":clean("assertive, educated, successful")
}


# Male, Female
# specifi race, and gender and get all test for that specific race and gender

# african_american_male , african_american_female
# african_american , european_american

data_terms = {
    'male_names': ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill'],
    'female_names': ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna'],
    'african_american_names': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha'],
    'european_american_names': ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Meredith', 'Sarah'],
    'african_american_female_names': ['Aisha', 'Ebony', 'Keisha', 'Lakisha', 'Latoya', 'Tamika', 'Imani', 'Shanice', 'Aaliyah', 'Precious', 'Nia', 'Deja', 'Latisha'],
    'african_american_male_names': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'DeShawn', 'DeAndre', 'Marquis', 'Terrell', 'Malik', 'Tyrone'],
    'european_american_male_names': ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Brad', 'Brendan', 'Geoffrey', 'Brett', 'Matthew', 'Neil'],
    'european_american_female_names': ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Molly', 'Amy', 'Claire', 'Katie', 'Madeline'],
    'male_terms': ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him'],
    'female_terms': ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her'],
    'european_american_terms': ['European American', 'British American', 'German American', 'Polish American', 'Russian American', 'Ukrainian American', 'Italian American', 'Portuguese American', 'French American', 'Romanian American', 'Greek American', 'Irish American', 'Spanish American', 'Bosnian American', 'Albanian American'],
    'african_american_terms': ['African American', 'Nigerian American', 'Ethiopian American', 'Egyptian American', 'Ghanaian American', 'Kenyan American', 'South African American', 'Somali American', 'Liberian American', 'Moroccan American', 'Cameroonian American', 'Cape Verdean American', 'Eritrean American', 'Sudanese American', 'Sierra Leonean American'],
    'european_american_female_terms': ['white female', 'white woman', 'white girl', 'fair-skinned woman'],
    'african_american_female_terms': ['black female', 'black woman', 'black girl', 'woman of color']
}


# src: Null It Out: Guarding Protected Attributes by Iterative NullspaceProjection

# src = https://arxiv.org/pdf/2003.11520.pdf
names = {
    'male_names': ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill'],
    'female_names': ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna'],
    'african_american' : [ 'alonzo', 'jamel', 'lerone', 'percell', 'theo', 'alphonse', 'jerome', 'leroy', 'rasaan',
                           'torrance', 'darnell', 'lamar', 'lionel', 'rashaun', 'tyree', 'lamont', 'malik',
                           'terrence', 'tyrone', 'everol', 'marcellus', 'terryl', 'wardell', 'aiesha',
                           'lashelle', 'nichelle', 'shereen', 'temeka', 'ebony', 'latisha', 'shaniqua', 'tameisha',
                           'teretha', 'jasmine', 'latonya', 'shanise', 'tanisha', 'tia', 'lakisha', 'latoya',
                           'tashika', 'yolanda', 'lashandra', 'malika', 'tawanda', 'yvette', 'hakim',
                           'jermaine', 'kareem', 'jamal', 'rasheed', 'aisha', 'keisha', 'kenya', 'tamika',
                           'shavonn',  'sharise', 'deion',  'lavon'],

    'european_american': ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin', 'ryan', 'andrew',
                           'fred', 'jack', 'matthew', 'stephen', 'brad', 'jed', 'todd', 'brandon',
                           'hank', 'jonathan', 'peter', 'wilbur', 'amanda', 'courtney', 'heather', 'melanie', 'sara',
                           'amber', 'crystal', 'katie', 'meredith', 'shannon', 'betsy', 'kristin', 'nancy',
                           'stephanie', 'bobbie-sue', 'ellen', 'lauren', 'peggy', 'sue-ellen', 'colleen', 'emily',
                           'megan', 'rachel', 'wendy', 'brendan', 'geoffrey', 'brett', 'jay', 'neil', 'anne', 'carrie',
                           'jill', 'laurie', 'kristen', 'paul', 'greg', 'sarah', 'donna']
}

# src: https://arxiv.org/pdf/2011.12086.pdf

new_concept = {
    'female_concept' : ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'aunt', 'mother', 'grandmother'],
    'male_concept' : ['male', 'man', 'boy', 'brother', 'he', 'him','his', 'son', 'uncle', 'father', 'grandfather'],
    'black_concept': ['Africa', 'Black', 'Jamaica', 'Haiti', 'Nigeria', 'Ethiopia', 'Somalia', 'Ghana', 'Barbados', 'Kenya', 'Liberia', 'Bahamas'],
                                                                                    'white_concept': ['German', 'Irish', 'English', 'Italian', 'Lebanese', 'Egyptian', 'Polish', 'French', 'Iranian', 'Slavic', 'Cajun', 'Chaldean']
    # Note "black" and "white" words to represent race are derived from https://2020census.gov/en/about-questions/2020-census-questions-race.html
}
# src: https://arxiv.org/pdf/2011.12086.pdf
new_names = {
    'male_names': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'DeShawn', 'DeAndre', 'Marquis', 'Terrell', 'Malik', 'Tyrone',
                   'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Jake', 'Connor', 'Tanner', 'Wyatt', 'Cody','Dustin'],
    'female_names': ['Allison',  'Anne',  'Carrie',  'Emily',  'Jill',  'Laurie',  'Kristen',  'Meredith',  'Molly',
                     'Amy',  'Claire', 'Madeline', 'Emma', 'Aisha', 'Keisha', 'Latonya', 'Lakisha', 'Latoya',
                     'Tamika', 'Imani', 'Shanice', 'Aaliyah', 'Nia', 'Latanya', 'Latisha', 'Deja'],
    'black_names': ['Aisha', 'Keisha', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Imani', 'Shanice', 'Aaliyah', 'Nia', 'Latanya', 'Latisha', 'Deja', 'Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'DeShawn', 'DeAndre', 'Marquis', 'Terrell', 'Malik', 'Tyrone'],
    'white_names': ['Allison',  'Anne',  'Carrie',  'Emily',  'Jill',  'Laurie',  'Kristen',  'Meredith', 'Molly',  'Amy',  'Claire', 'Madeline', 'Emma', 'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Jake', 'Connor', 'Tanner', 'Wyatt', 'Cody','Dustin']
}


"""
Copied from: https://github.com/thaleaschlender/An-Evaluation-of-Multiclass-Debiasing-Methods-on-Word-Embeddings/blob/main/An%20Evaluation%20of%20Multiclass%20Debiasing%20Methods%20on%20Word%20Embeddings(GloVe).ipynb
The 'religion' dictionary defines logical combinations for attribute sets. 
Each of these is defined for each combination of target sets. 
(Sets again are taken from Popovic et al.'s work (2020))
"""
religion_eval = {('islam_words', 'judaism_words'): [('likeable', 'unlikeable'), ('competent', 'incompetent'),
                                                    ('shy', 'aggressive'), ('intellectual_words', 'appearance_words'),
                                                    ('family', 'career'), ('instruments', 'weapons'),
                                                    ('pleasant', 'unpleasant'), ('science', 'art')],

                 ('islam_words', 'christianity_words'): [('likeable', 'unlikeable'), ('competent', 'incompetent'),
                                                    ('shy', 'aggressive'), ('intellectual_words', 'appearance_words'),
                                                    ('family', 'career'), ('instruments', 'weapons'),
                                                    ('pleasant', 'unpleasant'), ('science', 'art')],

                 ('judaism_words', 'christianity_words'): [('likeable', 'unlikeable'), ('competent', 'incompetent'),
                                                      ('shy', 'aggressive'), ('intellectual_words', 'appearance_words'),
                                                      ('family', 'career'), ('instruments', 'weapons'),
                                                      ('pleasant', 'unpleasant'), ('science', 'art')]
                 }

def get_religion_words(by: str = 'karve'):
    """
    :params: by: The set to use.
    :return:

    karve is  Manzini, Lim, Tsvetkov and Black (2019) equality sets. #@TODO: Check the validity of the comment.

    """

    if by.lower() == 'karve':
        return dict(judaism_words=['judaism', 'jew', 'jews', 'synagogue', 'synagogues', 'torah', 'rabbi', 'rabbis',
                                   'abraham', 'star', 'shabbat'],
                    christianity_words=['christianity', 'christian', 'christians', 'church', 'churches', 'bible',
                                        'priest',
                                        'priests',
                                        'jesus', 'cross', 'easter'],
                    islam_words=['islam', 'muslim', 'muslims', 'mosque', 'mosques', 'quran', 'imam', 'imams',
                                 'muhammad',
                                 'hilal', 'ramadan'])
    else:
        raise KeyError()



def get_weat_attribute_set():
    # Combine the two dictionary
    return combine_two_dictionaries(dict1, dict2)


def get_gender_definitional():
    return [['woman', 'man'],
            ['girl', 'boy'],
            ['she', 'he'],
            ['mother', 'father'],
            ['daughter', 'son'],
            ['gal', 'guy'],
            ['female', 'male'],
            ['her', 'his'],
            ['herself', 'himself'],
            ['mary', 'john']]  # Source: tolga


def get_eq_pairs():
    return [
        ["monastery", "convent"],
        ["spokesman", "spokeswoman"],
        ["Catholic_priest", "nun"],
        ["Dad", "Mom"],
        ["Men", "Women"],
        ["councilman", "councilwoman"],
        ["grandpa", "grandma"],
        ["grandsons", "granddaughters"],
        ["prostate_cancer", "ovarian_cancer"],
        ["testosterone", "estrogen"],
        ["uncle", "aunt"],
        ["wives", "husbands"],
        ["Father", "Mother"],
        ["Grandpa", "Grandma"],
        ["He", "She"],
        ["boy", "girl"],
        ["boys", "girls"],
        ["brother", "sister"],
        ["brothers", "sisters"],
        ["businessman", "businesswoman"],
        ["chairman", "chairwoman"],
        ["colt", "filly"],
        ["congressman", "congresswoman"],
        ["dad", "mom"],
        ["dads", "moms"],
        ["dudes", "gals"],
        ["ex_girlfriend", "ex_boyfriend"],
        ["father", "mother"],
        ["fatherhood", "motherhood"],
        ["fathers", "mothers"],
        ["fella", "granny"],
        ["fraternity", "sorority"],
        ["gelding", "mare"],
        ["gentleman", "lady"],
        ["gentlemen", "ladies"],
        ["grandfather", "grandmother"],
        ["grandson", "granddaughter"],
        ["he", "she"],
        ["himself", "herself"],
        ["his", "her"],
        ["king", "queen"],
        ["kings", "queens"],
        ["male", "female"],
        ["males", "females"],
        ["man", "woman"],
        ["men", "women"],
        ["nephew", "niece"],
        ["prince", "princess"],
        ["schoolboy", "schoolgirl"],
        ["son", "daughter"],
        ["sons", "daughters"],
        ["twin_brother", "twin_sister"]
    ]


# TODO: add more stuff here as get_relegion words are expanded

temp_database = {
    'male_names': "John, Paul, Mike, Kevin, Steve, Greg, Jeff, Bill",
    'female_names': "Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna",
    'male_concept': ['male', 'man', 'boy', 'brother', 'he', 'him','his', 'son', 'uncle', 'father', 'grandfather'],
    'female_concept': ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'aunt', 'mother', 'grandmother'],
    'european_american_names': "Brad, Brendan, Geoffrey, Greg, Brett, Matthew, Neil, Todd, Allison, Anne, Carrie,Emily, Jill, Laurie, Meredith, Sarah",
    'african_american_names': "Darnell, Hakim, Jermaine, Kareem, Jamal, Leroy, Rasheed, Tyrone, Aisha, Ebony,Keisha, Kenya, Lakisha, Latoya, Tamika, Tanisha",
    'european_american_concept': ['German', 'Irish', 'English', 'Italian', 'Lebanese', 'Egyptian', 'Polish', 'French', 'Iranian', 'Slavic', 'Cajun', 'Chaldean'],
    'african_american_concept': ['Africa', 'Black', 'Jamaica', 'Haiti', 'Nigeria', 'Ethiopia', 'Somalia', 'Ghana', 'Barbados', 'Kenya', 'Liberia', 'Bahamas'],
    'african_american_female_names': "Aisha, Keisha, Lakisha, Latisha, Latoya, Malika, Nichelle,Shereen, Tamika, Tanisha, Yolanda, Yvette",
    'african_american_male_names': "Alonzo, Alphonse, Hakim, Jamal, Jamel, Jerome, Leroy, Lionel,Marcellus, Terrence, Tyrone, Wardell",
    'european_american_female_names': "Carrie,  Colleen,  Ellen,  Emily,  Heather,  Katie,  Megan,Melanie, Nancy, Rachel, Sarah, Stephanie",
    'european_american_male_names': "Andrew, Brad, Frank, Geoffrey, Jack, Jonathan, Josh, Matthew,Neil, Peter, Roger, Stephen"
}




def clean_data(database):
    for key, value in database.items():
        if type(value) == str:
            database[key] = clean(value)
    return database


database = clean_data(temp_database)

database['all_male_names'] = database['european_american_male_names'] + database['african_american_male_names']
database['all_female_names'] = database['european_american_female_names'] + database['african_american_female_names']

combined_db = combine_two_dictionaries(combine_two_dictionaries(get_weat_attribute_set(), get_religion_words(by='karve')), names)
database['male'] = dict2['male']
database['female'] = dict2['female']

database['male_2'] = dict2['male_2']
database['female_2'] = dict2['female_2']

def get_associated_words(word):
    return dict2[word]

