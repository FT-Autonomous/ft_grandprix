# Install

First install the necessary requirements:

```
pip3 install -r requirements.txt
```

# Image Splitting

This simulator loads in maps as heightmaps.
Mujoco heightmaps are documented [here](https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-hfield).
For performance reasons heigtmaps cannot be loaded in full into the sim, as this severly impacts rangefinder speed.
For this reason, heightmaps have to be split into multiple smaller heightmaps and sections that do not contribute anything to the final map must be deleted.
This is the purpose of the `ft_grandprix.chunk` utility.

To split a heightmap, invoke the following (`IMAGE` can be set to any
track. See the *Tracks*).

```
python3 -m ft_grandprix.chunk -f -i IMAGE
```

You are able to configure the chunk width and height in pixels (invoke `python3 -m ft_grandprix.chunk -h` for more details).

# Environment Generation

The environmet refers to a combination of the map image being used and the cars contained within the map.
The environment is auto generated using empy.
Empy is documented [here](https://ecell3.readthedocs.io/en/latest/empy-manual.html).
It is a templating language that will populate an MJCF file with the appropriate heigtfield `geom` entities based on the contents of the `rendered/chunks` directory and the `template/cars.json` file.

To create the environment, invoke:

```
python3 -m ft_grandprix.map
```

## cars.json

The `template/cars.json` file format is an array of objects containing the following fields.

- `primary` - primary color. can be set to one of the colors in `ft_grandprix.colors` or `random`, in which case a random color will be chosen.
- `secondary` - same semantics as `primary`
- `icon` a *square* PNG image to use as an icon on the car. These images should be stored in the `template/icons/` directory.

# Running the Simulator

*NOTE*: See special instructions for MacOS at the bottom.

You can then run the simulator as follows:

```
python3 -m ft_grandprix.drive
```

## MacOS

MacOS users should use the command:

```
mjpython -m ft_grandprix.drive
```

# Overall Commands

```
python3 -m ft_grandprix.chunk -i template/track.png -f
python3 -m ft_grandprix.map --rangefinders 90
python3 -m ft_grandprix.drive
```

# Custom Keybindings

In addition to the existing mujoco keybindings, I have defined
additional keybindings, namely:

- `N` - focus on next car
- `P` - focus on previous car
- `L` - liberate camera

![an image of the simulator](images/ft_grandprix_volume_2.png)
