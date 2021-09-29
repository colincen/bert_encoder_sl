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


father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'F': ['entity_name', 'playlist', 'artist', 'timeRange', 'party_size_description', 'poi', 'restaurant_name', 'album', 'track', 'object_name', 'movie_name'],
 'A': ['playlist_owner', 'spatial_relation', 'condition_temperature', 'object_select', 'rating_value'], 
 'E': ['music_item', 'state', 'restaurant_type', 'sort', 'country', 'cuisine', 'facility', 'current_location', 'condition_description', 'year', 'object_part_of_series_type'], 
 'D': ['party_size_number', 'best_rating'], 
 'B': ['city', 'served_dish', 'service', 'genre', 'object_location_type', 'location_name'], 
 'C': ['geographic_poi'], 
 'G': ['object_type', 'rating_unit', 'movie_type']
 }

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'E': ['entity_name', 'timeRange', 'party_size_description', 'geographic_poi', 'track', 'object_name'],
 'D': ['playlist', 'artist', 'city', 'served_dish', 'poi', 'restaurant_name', 'album', 'service', 'movie_name'],
  'C': ['playlist_owner', 'music_item', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'facility', 'current_location', 'condition_temperature', 'condition_description', 'year', 'genre', 'object_select', 'object_type', 'object_part_of_series_type'], 
  'F': ['party_size_number', 'rating_value', 'best_rating'], 
  'G': ['country', 'cuisine'], 
  'A': ['rating_unit'], 
  'B': ['movie_type', 'object_location_type', 'location_name']
  }
father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['current_location', 'condition_temperature', 'movie_name', 'cuisine'],
 'B': ['rating_value', 'condition_description', 'restaurant_name', 'object_location_type', 'best_rating'],
  'C': ['city', 'artist', 'state', 'served_dish', 'party_size_number'],
   'D': ['geographic_poi', 'facility', 'poi', 'playlist_owner', 'party_size_description', 'genre', 'object_type', 'object_name'],
    'E': ['music_item', 'country', 'spatial_relation', 'object_part_of_series_type'], 
    'F': ['entity_name', 'movie_type', 'rating_unit', 'playlist'],
     'G': ['year', 'location_name', 'album', 'sort', 'restaurant_type', 'object_select', 'service', 'track', 'timeRange']
     }

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['timeRange', 'object_part_of_series_type', 'cuisine', 'music_item', 'object_type', 'object_location_type'], 
'B': ['condition_description', 'party_size_number', 'entity_name', 'served_dish', 'artist', 'party_size_description'],
 'C': ['album', 'playlist_owner', 'condition_temperature'], 
 'D': ['track', 'genre', 'state', 'rating_value', 'city', 'object_select'], 
 'E': ['playlist', 'movie_name'], 
 'F': ['movie_type', 'restaurant_type', 'geographic_poi', 'spatial_relation', 'object_name', 'location_name', 'current_location', 'poi', 'restaurant_name'], 
 'G': ['facility', 'service', 'sort', 'best_rating', 'country', 'year', 'rating_unit']
 }

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['sort', 'timeRange', 'condition_description', 'object_select', 'geographic_poi'], 
'B': ['country', 'album', 'city', 'best_rating', 'object_part_of_series_type', 'spatial_relation', 'rating_unit'], 
'C': ['party_size_description', 'movie_name', 'track', 'state'], 
'D': ['playlist', 'cuisine', 'movie_type'], 
'E': ['poi', 'music_item', 'current_location', 'restaurant_name', 'entity_name', 'service', 'year', 'served_dish', 'object_type', 'artist', 'object_location_type', 'condition_temperature', 'party_size_number', 'genre', 'restaurant_type'], 
'F': ['rating_value', 'facility', 'playlist_owner'], 
'G': ['location_name', 'object_name']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['movie_type', 'entity_name', 'sort', 'cuisine', 'rating_value', 'restaurant_name', 'party_size_number'], 
'B': ['track', 'geographic_poi', 'playlist', 'served_dish', 'object_part_of_series_type', 'poi'], 
'C': ['genre', 'party_size_description', 'playlist_owner', 'city', 'music_item'], 
'D': ['facility', 'current_location'],
 'E': ['condition_description', 'timeRange', 'spatial_relation', 'object_type'], 
 'F': ['object_name', 'object_location_type', 'object_select', 'state', 'artist', 'restaurant_type', 'service'],
  'G': ['location_name', 'year', 'rating_unit', 'movie_name', 'condition_temperature', 'best_rating', 'country', 'album']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['entity_name', 'playlist', 'artist', 'timeRange', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'cuisine', 'geographic_poi', 'album', 'track', 'genre', 'object_name', 'movie_name', 'location_name'], 
'B': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'facility', 'current_location', 'condition_temperature', 'condition_description', 'service', 'year', 'object_select', 'rating_value', 'best_rating', 'object_part_of_series_type'],
 'C': ['country', 'object_type', 'rating_unit', 'movie_type', 'object_location_type']
 }


father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['entity_name', 'playlist', 'artist', 'timeRange', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'cuisine', 'geographic_poi', 'album', 'service', 'track', 'genre', 'object_name', 'movie_name', 'location_name'],
'B': ['playlist_owner', 'party_size_number', 'sort', 'spatial_relation', 'current_location', 'condition_temperature', 'condition_description', 'year', 'rating_value', 'best_rating', 'object_part_of_series_type'],
'C': ['music_item', 'state', 'restaurant_type', 'country', 'facility', 'object_select', 'object_type', 'rating_unit', 'movie_type', 'object_location_type']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['entity_name', 'playlist', 'artist', 'timeRange', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'geographic_poi', 'album', 'track', 'genre', 'object_name', 'movie_name', 'location_name'], 
'C': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'current_location', 'condition_temperature', 'condition_description', 'service', 'year', 'object_select', 'rating_value', 'best_rating', 'object_part_of_series_type'],
'B': ['country', 'cuisine', 'facility', 'object_type', 'rating_unit', 'movie_type', 'object_location_type']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'B': ['entity_name', 'playlist', 'artist', 'timeRange', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'geographic_poi', 'album', 'track', 'object_name', 'movie_name', 'location_name'], 
'A': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'restaurant_type', 'sort', 'spatial_relation', 'cuisine', 'current_location', 'condition_temperature', 'condition_description', 'service', 'year', 'genre', 'object_select', 'rating_value', 'best_rating', 'object_part_of_series_type'], 
'C': ['country', 'facility', 'object_type', 'rating_unit', 'movie_type', 'object_location_type']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['object_part_of_series_type', 'timeRange', 'genre', 'playlist_owner', 'geographic_poi', 'country', 'served_dish', 'track', 'artist', 'best_rating', 'object_select', 'object_location_type', 'condition_description', 'location_name', 'spatial_relation', 'rating_unit', 'movie_name'], 
'B': ['facility', 'restaurant_name', 'entity_name', 'movie_type', 'cuisine', 'service', 'album', 'restaurant_type', 'party_size_description', 'sort', 'music_item'], 
'C': ['object_name', 'city', 'poi', 'party_size_number', 'current_location', 'playlist', 'state', 'condition_temperature', 'year', 'rating_value', 'object_type']
}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['condition_description', 'party_size_number', 'track', 'playlist', 'cuisine', 'location_name', 'movie_name', 'rating_value', 'timeRange', 'object_location_type', 'movie_type', 'party_size_description', 'served_dish', 'spatial_relation'], 
'B': ['album', 'entity_name', 'sort', 'artist', 'service', 'object_select', 'city', 'state', 'playlist_owner', 'restaurant_name', 'object_name', 'year'], 
'C': ['condition_temperature', 'best_rating', 'facility', 'country', 'current_location', 'music_item', 'geographic_poi', 'poi', 'restaurant_type', 'genre', 'object_part_of_series_type', 'rating_unit', 'object_type']
}
father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['sort', 'object_type', 'spatial_relation', 'city', 'current_location', 'facility', 'served_dish', 'best_rating', 'party_size_description'], 'B': ['service', 'movie_name', 'playlist', 'geographic_poi', 'entity_name', 'object_name', 'object_select', 'poi', 'track', 'object_part_of_series_type', 'rating_value', 'rating_unit', 'year', 'condition_description', 'playlist_owner', 'country', 'timeRange', 'object_location_type'], 'C': ['music_item', 'movie_type', 'party_size_number', 'condition_temperature', 'genre', 'location_name', 'cuisine', 'state', 'restaurant_type', 'album', 'artist', 'restaurant_name']}

father_son_slot = {
'pad':['<PAD>'],
'O':['O'],
'A': ['entity_name', 'track', 'object_location_type', 'condition_description', 'movie_type', 'country', 'poi', 'party_size_description', 'movie_name', 'rating_value', 'sort', 'party_size_number', 'object_name', 'object_part_of_series_type', 'location_name', 'object_type', 'state', 'timeRange'], 
'B': ['album', 'music_item', 'rating_unit', 'service', 'artist', 'geographic_poi', 'facility', 'current_location', 'condition_temperature', 'year', 'city'], 
'C': ['genre', 'best_rating', 'served_dish', 'restaurant_type', 'playlist_owner', 'playlist', 'cuisine', 'spatial_relation', 'object_select', 'restaurant_name']}
