############################### five ######################################
# 1
# father_son_slot = {
#                     'pad':['<PAD>'],
#                 'O':['O'],
#     'A': ['entity_name', 'party_size_description', 'restaurant_name', 'geographic_poi', 
# 'object_name'],
#  'C': ['playlist', 'artist', 'timeRange', 'city', 'served_dish', 'poi', 
# 'album', 'service', 'track', 'movie_name', 'object_location_type', 'location_name'], 
# 'D': ['playlist_owner', 'music_item', 'state', 'restaurant_type', 'sort', 'spatial_relation', 
# 'cuisine', 'facility', 'current_location', 'condition_temperature', 'condition_description', 
# 'year', 'genre', 'object_select', 'object_type', 'object_part_of_series_type', 
# 'rating_unit'],
#  'B': ['party_size_number', 'rating_value', 'best_rating'], 
#  'E': ['country', 'movie_type']}

# 2
# father_son_slot={
# 'pad':['<PAD>'],
# 'O':['O'],
# 'D': ['entity_name', 'party_size_description', 'restaurant_name', 'geographic_poi', 'album', 'track', 'object_name', 'movie_name'],
#  'A': ['playlist', 'artist', 'timeRange', 'city', 'served_dish', 'poi', 'service', 'genre', 'object_location_type', 'location_name'], 
#  'E': ['playlist_owner', 'music_item', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'cuisine', 'facility', 'current_location', 'condition_temperature', 'condition_description', 'year', 'object_select', 'object_type', 'object_part_of_series_type'], 
#  'B': ['party_size_number', 'rating_value', 'best_rating'], 
#  'C': ['country', 'rating_unit', 'movie_type']} 

# 3
# father_son_slot ={
#     'pad':['<PAD>'],
#     'O':['O'],
#     'C': ['entity_name', 'playlist', 'timeRange', 'party_size_description', 'poi', 'restaurant_name', 'geographic_poi', 'album', 'track', 'object_name', 'movie_name'],
#      'B': ['artist', 'restaurant_type', 'city', 'served_dish', 'cuisine', 'service', 'genre', 'location_name'], 
#      'D': ['playlist_owner', 'state', 'sort', 'spatial_relation', 'country', 'facility', 'current_location', 'condition_temperature', 'condition_description', 'year', 'object_select'], 
#      'E': ['music_item', 'object_type', 'object_part_of_series_type', 'movie_type', 'object_location_type'], 
#      'A': ['party_size_number', 'rating_value', 'best_rating', 'rating_unit']}


# 4 random
# father_son_slot = {
# 'pad':['<PAD>'],
# 'O':['O'],
# 'A': ['service', 'condition_description', 'geographic_poi', 'rating_value', 'cuisine', 'rating_unit'], 
# 'B': ['music_item', 'spatial_relation', 'best_rating', 'facility', 'object_part_of_series_type', 'country', 'city'], 
# 'C': ['genre', 'party_size_description', 'playlist', 'object_name', 'party_size_number', 'album', 'restaurant_type', 'movie_name'], 
# 'D': ['playlist_owner', 'object_location_type', 'year', 'poi', 'object_type', 'condition_temperature', 'entity_name', 'track', 'current_location', 'restaurant_name', 'artist'], 
# 'E': ['served_dish', 'location_name', 'sort', 'object_select', 'timeRange', 'state', 'movie_type']
# }

#5 random
# father_son_slot = {
# 'pad':['<PAD>'],
# 'O':['O'],
#  'A': ['condition_description', 'restaurant_name', 'timeRange', 'object_select', 'restaurant_type', 'location_name'], 
#  'B': ['spatial_relation', 'party_size_description', 'sort', 'condition_temperature', 'best_rating', 'service', 'album', 'country', 'geographic_poi'], 
#  'C': ['artist', 'served_dish', 'rating_value', 'rating_unit', 'poi', 'music_item', 'object_location_type', 'object_name', 'movie_name', 'object_type'], 
#  'D': ['object_part_of_series_type', 'facility', 'entity_name', 'track', 'state', 'genre', 'year', 'cuisine', 'playlist'], 
#  'E': ['current_location', 'city', 'party_size_number', 'playlist_owner', 'movie_type']}

# # 6
# father_son_slot = {
#     'pad':['<PAD>'],
#     'O':['O'],
#     'D': ['entity_name', 'timeRange', 'party_size_description', 'geographic_poi', 'album', 'track', 'object_name', 'movie_name'], 
#     'B': ['playlist', 'artist', 'city', 'served_dish', 'poi', 'restaurant_name', 'service', 'genre', 'object_location_type', 'location_name'],
#      'E': ['playlist_owner', 'music_item', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'facility', 'current_location', 'condition_temperature', 'condition_description', 'year', 'object_select', 'object_type', 'object_part_of_series_type'], 
#      'A': ['party_size_number', 'rating_value', 'best_rating'], 
#      'C': ['country', 'cuisine', 'rating_unit', 'movie_type']}

#  random 
# father_son_slot = {
# 'pad':['<PAD>'],
# 'O':['O'],
# 'A': ['sort', 'poi', 'track', 'cuisine', 'rating_unit', 'object_select'], 
# 'B': ['geographic_poi', 'city', 'served_dish', 'restaurant_name', 'genre'], 
# 'C': ['year', 'location_name', 'best_rating', 'playlist', 'condition_description'], 
# 'D': ['facility', 'timeRange', 'movie_name', 'album', 'spatial_relation', 'service', 'party_size_number', 'object_location_type'], 
# 'E': ['music_item', 'condition_temperature', 'object_type', 'object_name', 'playlist_owner', 'object_part_of_series_type', 'party_size_description', 'movie_type', 'country', 'entity_name', 'rating_value', 'restaurant_type', 'state', 'artist', 'current_location']}



############################### seven ####################################

father_son_slot = {
    'pad':['<PAD>'],
    'O':['O'],
    'G': ['entity_name', 'timeRange', 'party_size_description', 'album', 'track', 'object_name', 'movie_name'], 
    'C': ['playlist', 'artist', 'city', 'served_dish', 'poi', 'restaurant_name', 'cuisine', 'genre', 'location_name'],
    'B': ['playlist_owner', 'music_item', 'state', 'sort', 'spatial_relation', 'current_location', 'condition_temperature', 'condition_description', 'service', 'year', 'object_select', 'object_part_of_series_type'], 
    'D': ['party_size_number', 'rating_value', 'best_rating'], 
    'E': ['restaurant_type', 'object_type', 'rating_unit', 'movie_type', 'object_location_type'], 
    'A': ['country', 'facility'], 
    'F': ['geographic_poi']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'F': ['entity_name', 'playlist', 'artist', 'timeRange', 'party_size_description', 'album', 'track', 'object_name', 'movie_name'], 
'A': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'sort', 'spatial_relation', 'current_location', 'condition_temperature', 'condition_description', 'year', 'object_select', 'rating_value', 'object_part_of_series_type'],
 'B': ['restaurant_type', 'city', 'served_dish', 'poi', 'restaurant_name', 'cuisine', 'service', 'genre', 'object_location_type', 'location_name'], 
 'G': ['country', 'facility', 'object_type', 'rating_unit', 'movie_type'], 
 'C': ['geographic_poi'], 
 'D': ['best_rating']}