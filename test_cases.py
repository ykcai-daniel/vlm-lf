#for video hong_kong_airport.mp4
#from 8:10 to 9:40 (90s) (hong_kong_airport_demo_data.mp4)
#Resample the video at 5fps, frame index starts from zero. The intervals are an array of tuple of (start,end)
test_cases=[
    {
        'object':'black and white striped backpack',
        'intervals': [('01:10', '01:15')],
    },
    {
        'object': 'black backpack',
        'intervals': [('00:17', '00:19'), ('00:23', '00:24'), ('00:49', '00:50'), ('00:59', '01:14')],
    },
    {
        'object':'white suitcase',
        'intervals': [('00:11', '00:13')],
    },
    {
        'object': 'tote bag',
        'intervals': [('00:17', '00:19'), ('00:31', '00:34'), ('00:55', '00:57'), ('01:37', '01:30')],
    },
    {
        'object': 'pink_short_luggage.jpg',
        'intervals': [('00:17', '00:21')],
    },
    {
        'object': 'silver_luggage.jpg',
        'intervals': [('01:17', '01:25')],
    },
    {
        'object':'white_shirt_blue_jeans_man.jpg',
        'intervals':[('00:07', '00:20')],
    },
    {
        'object': 'suitcase_mask_man.jpg',
        'intervals': [('01:28', '01:32')],
    },
    {
        'object': 'rainbow_shirt_white_shoes_girl.jpg',
        'intervals': [('00:15', '00:21')],
    },
    {
        'object': 'woman_with_phone_trolley.jpg',
        'intervals': [('00:36', '00:44')],
    },

    {
        # negative sample: should not exist in any frame
        'object': 'steak',
        'intervals': [],
    }

    # two more novel luggages (described with color/shape)
    # two mans and two women (one with plain clothing and the other with fancier clothing)
    # images of man and woman and the novel luggages placed in test_images
]