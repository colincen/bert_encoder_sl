import random
slot_list = ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist','city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
    'poi', 'sort', 'spatial_relation', 'state', 'party_size_description','city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description', 'genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist',
'object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating',
'object_name', 'object_type', 'timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
slot_list = list(set(slot_list))
mp = {1:'A',2:'B',3:'C',4:'D',5:'E'}
res_dict = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
for i, slot in enumerate(slot_list):
    t = random.randint(1,5)
    res_dict[mp[t]].append(slot)
print(res_dict)