#for video chunk 3 
#from 8:00 to 9:40 (100s)
#Resample the video at 5fps, frame index starts from zero. The intervals are an array of tuple of (start,end)
test_cases=[
    {
        'query':'white backpack',
        'intervals':[],
    },
    {
        'object':'white suitcase',
        'intervals':[],
    },
    {
        #negative sample: should not exist in any frame
        'object':'steak',
        'intervals':[],
    },
    {
        'object':'black backpack',
        'intervals':[],
    },

    {
        'object':'tote bag',
        'intervals':[],
    },
    {
        'object':'white_shirt_blue_jeans_man.jpg',
        'intervals':['00:17','00:20'],
    }
    # two more novel luggages (described with color/shape)
    # two mans and two women (one with plain clothing and the other with fancier clothing)
    #images of man and woman and the novel luggages placed in test_images
]