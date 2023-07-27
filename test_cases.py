#for video chunk 3 
#from 8:00 to 9:40 (100s)
#Resample the video at 6fps (5 frames per second), frame index starts from zero. The intervals are an array of tuples
test_cases=[
    {
        'query':'white backpack',
        'intervals':[],
    },
    {
        'object':'white suitcase'
    },
    {
        #negative sample: should not exist in any frame
        'object':'steak'
    },
    {
        'object':'black_backpack'
    },

    {
        'object':'tote bag'
    }
    # two more novel luggages (described with color/shape)
    # two mans and two women (one with plain clothing and the other with fancier clothing)
    #images of man and woman and the novel luggages placed in test_images
]