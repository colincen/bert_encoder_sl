import spacy
nlp = spacy.load('en_core_web_sm')
snips_data = {
    "music_item": ["song", "track"],
    "playlist_owner": ["my", "donna s"],
    "entity_name": ["the crabfish", "natasha"],
    "playlist": ["quiero playlist", "workday lounge"],
    "artist": ["lady bunny", "lisa dalbello"],
    "city": ["north lima", "falmouth"],
    "facility": ["smoking room", "indoor"],
    "timeRange": ["9 am", "january the twentieth"],
    "restaurant_name": ["the maisonette", "robinson house"],
    "country": ["dominican republic", "togo"],
    "cuisine": ["ouzeri", "jewish"],
    "restaurant_type": ["tea house", "tavern"],
    "served_dish": ["wings", "cheese fries"],
    "party_size_number": ["seven", "one"],
    "poi": ["east brady", "fairview"],
    "sort": ["top-rated", "highly rated"], 
    "spatial_relation": ["close", "faraway"],
    "state": ["sc", "ut"],
    "party_size_description": ["me and angeline", "my colleague and i"],
    "current_location": ["current spot", "here"],
    "geographic_poi": ["bashkirsky nature reserve", "narew national park"],
    "condition_temperature": ["chillier", "hot"],
    "condition_description": ["humidity", "depression"],
    "genre": ["techno", "pop"],
    "service": ["spotify", "groove shark"],
    "year": ["2005", "1993"],
    "album": ["allergic", "secrets on parade"],
    "track": ["in your eyes", "the wizard and i"],
    "object_part_of_series_type": ["series", "saga"],
    "object_select": ["this", "current"],
    "rating_value": ["1", "four"],
    "object_name": ["american tabloid", "my beloved world"],
    "object_type": ["book", "novel"],
    "rating_unit": ["points", "stars"],
    "best_rating": ["6", "5"],
    "movie_type": ["animated movies", "films"],
    "object_location_type": ["movie theatre", "cinema"],
    "location_name": ["amc theaters", "wanda group"],
    "movie_name": ["on the beat", "for lovers only"]
}
woz_data = {"depart": ["arbury lodge guesthouse", "the cambridge corn exchange"],
    "day": ["friday","monday"],
    "people" : ["5","seven"],
    "leave": ["after 11:00", "before 14:00"],
    "arrive": ["14:16", "11:45"],
    "dest": ["regency gallery", "maharajah tandoori restaurant"],
    "price": ["cheaper", "expensive"],
    "area": ["eastside", "same area and price"],
    "stay":["1", "eight"],
    "stars":["0", "1"],
    "food":["portuguese", "thai and chinese"],
    "name": ["ashley hotel", "acorn guest house"],
    "time": ["18:00","17:15"],
    "type": ["cinemas","college"]
}
class coarse_fine:
    def __init__(self, dataset, type):
        if type == 'pos':
            coarse, bins_labels, father2son = self.buildfrompos(dataset)
        if type == 'bert_5_3':
            coarse, bins_labels, father2son = self.buildfrombertreps_53(dataset)
        if type == 'bert_7_4':
            coarse, bins_labels, father2son = self.buildfrombertreps_74(dataset)

        self.coarse, self.bins_labels, self.father2son = coarse, bins_labels, father2son

    def buildfrompos(self, dataset):
        if dataset == 'snips':
            father_son_slot={
                'pad':['<PAD>'],
                'O':['O'],
                'NUM':['timeRange','party_size_number', 'year'],
                'X':['state','rating_value','best_rating'],
                'ADJADV':['spatial_relation','cuisine', 'facility', 'condition_temperature', 'object_select','sort','movie_name','current_location'],
                'PROPN':['music_item','playlist_owner','entity_name','playlist','artist','city','restaurant_name','country','poi','location_name','geo','geographic_poi','album','track','object_part_of_series_type','object_name'],
                'NOUN':['restaurant_type','served_dish','party_size_description','rating_unit','genre','condition_description','object_type','object_location_type','movie_type'],
                'VERB':['sort','service']
            }
            bins_labels = ['pad','O','B-NUM','I-NUM','B-X','I-X','B-ADJADV','I-ADJADV','B-PROPN','I-PROPN','B-NOUN','I-NOUN','B-VERB','I-VERB']
            coarse = ['pad','O','NUM','X','ADJADV','PROPN','NOUN','VERB']

        if dataset == 'multiwoz':
            father_son_slot = {
                'pad':['<PAD>'],
                'O':['O'],
                'NUM':['people','leave','arrive','stay','time','stars'],
                'PROPN':['day','depart','name','food'],
                'NOUN':['dest','area','type'],
                'ADJ':['price']
            }
            bins_labels = ['pad','O','B-NUM','I-NUM','B-PROPN','I-PROPN','B-NOUN','I-NOUN','B-ADJ','I-ADJ']
            coarse = ['pad','O','NUM','PROPN','NOUN','ADJ']

        return coarse, bins_labels, father_son_slot

    def buildfrombertreps_53(self, dataset):
        if dataset == 'snips':
            father_son_slot={
                'pad':['<PAD>'],
                'O':['O'],
                'A': ['entity_name', 'playlist', 'artist', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'album', 'track', 'object_name', 'movie_name'], 
                'B': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'spatial_relation', 'current_location', 'condition_temperature', 'year', 'genre', 'object_select', 'rating_value', 'object_part_of_series_type'], 
                'C': ['restaurant_type', 'sort', 'cuisine', 'facility', 'condition_description', 'service', 'object_type', 'movie_type', 'location_name'], 
                'D': ['timeRange', 'country', 'geographic_poi'], 
                'E': ['best_rating', 'rating_unit', 'object_location_type']
          
            }
            bins_labels = ['pad','O','B-A','I-A','B-B','I-B','B-C','I-C','B-D','I-D','B-E','I-E']
            coarse = ['pad','O','A','B','C','D','E']

        if dataset == 'multiwoz':
            father_son_slot = {
                'pad':['<PAD>'],
                'O':['O'],
                'A':['dest','depart','day','price','type','name','food'],
                'B':['arrive','leave','time'],
                'C':['people','stay','stars','area']

            }
            bins_labels = ['pad','O','B-A','I-A','B-B','I-B','B-C','I-C']
            coarse = ['pad','O','A','B','C']

        return coarse, bins_labels, father_son_slot

    def buildfrombertreps_74(self, dataset):
        if dataset == 'snips':
            father_son_slot={
                'pad':['<PAD>'],
                'O':['O'],
                'A': ['entity_name', 'timeRange', 'party_size_description', 'album', 'track', 'object_name', 'movie_name'], 
                'B': ['playlist', 'artist', 'city', 'served_dish', 'poi', 'restaurant_name', 'genre', 'object_location_type', 'location_name'], 
                'C': ['playlist_owner', 'sort', 'spatial_relation', 'current_location', 'condition_temperature', 'service', 'year', 'object_select', 'object_part_of_series_type'],
                'D': ['music_item', 'state', 'restaurant_type', 'country', 'cuisine', 'facility', 'condition_description', 'object_type', 'rating_unit'], 
                'E': ['party_size_number', 'rating_value', 'best_rating'], 
                'F': ['geographic_poi'], 
                'G': ['movie_type']          
            }
            bins_labels = ['pad','O','B-A','I-A','B-B','I-B','B-C','I-C','B-D','I-D','B-E','I-E','B-F','I-F','B-G','I-G']
            coarse = ['pad','O','A','B','C','D','E','F','G']


        if dataset == 'multiwoz':
            father_son_slot = {
                'pad':['<PAD>'],
                'O':['O'],
                'A': ['dest', 'depart'], 
                'B': ['arrive', 'leave', 'time'], 
                'C': ['day', 'price', 'type', 'area', 'name', 'food'], 
                'D': ['people', 'stay', 'stars']
            }
            bins_labels = ['pad','O','B-A','I-A','B-B','I-B','B-C','I-C','B-D','I-D']
            coarse = ['pad','O','A','B','C','D']

        return coarse, bins_labels, father_son_slot



# def gen_ner(slot2example):
#     res = {}
#     for k, v in slot2example.items():
#         res[k] = []
#         for e in v:
#             doc = nlp(e)
#             temp = []
#             for i in doc.ents:
#                 temp.append(i.label_)
#             res[k].append(temp)
#     return res


# d = gen_ner(snips_data)
# for k,v in d.items():
#     print(k)
#     print(v)
#     print('-'*10)
