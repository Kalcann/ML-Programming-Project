import pathlib
from tkinter import filedialog as fd

import NetworkManager as nm

data_dir = pathlib.Path('TrainingData')

guitars = list(data_dir.glob('Guitar\\*'))
drums = list(data_dir.glob('Drums\\*'))
pianos = list(data_dir.glob('Piano\\*'))


print('\ndecompiling dataset\n')
data_set = nm.DataSet([guitars, drums, pianos], ('Guitar', 'Drum', 'Piano'))
data_set.decompile_data()


loaded_net = None

loop = True

while loaded_net is None:
    user_input = input('\nLoad(l) ... Create(c))\n option: ')

    if user_input == 'c':
        name = input('network name: ')
        loaded_net = nm.Network(
            name,  # name
            100,  # depth
            50,  # width
            3,  # input width
            3,  # output width
            0.5,  # min weight
            1.7,  # max weight
            -50,  # min bias
            40  # max bias
        )

    elif user_input == 'l':
        print('minimize window if you cannot find file dialogue')
        try:
            img = fd.askopenfilename()
            loaded_net = nm.load(img)
        except:
            print('error loading network')

while loop:

    user_input = input('\nTrain(t) ... Save(s) ... predict(p) ... quit(q)\nOption: ')

    if user_input == 't':
        gens = int(input('how many generations: '))
        nm.train(
            loaded_net,
            20,
            5,
            gens,
            data_set,
            10,
            50,
            0.50
        )
    elif user_input == 's':
        loaded_net.save()

    elif user_input == 'p':
        print('minimize window if you cannot find file dialogue')
        try:
            img = fd.askopenfilename()
            result = loaded_net.predict(img)
            print(f'\n\nI think it is a {data_set.tags[result[0]]}\nCertainty: {result[1]}')
        except:
            print('error loading image')
    elif user_input == 'q':
        loop = False


