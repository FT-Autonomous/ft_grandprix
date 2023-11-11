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
This is the purpose of the ~ft_grandprix.chunk~ utility.

To split a heightmap, invoke:

```
python3 -m ft_grandprix.chunk -i IMAGE
```

# Environment Generation

The environment is generated based off the chunks using empy.
Empy is documented [here](https://ecell3.readthedocs.io/en/latest/empy-manual.html).
It is a templating language that will populate an MJCF file with the appropriate heigtfield `geom` entities based on the contents of the `chunks` directory.

To create the environment, invoke:

```
python3 -m em car.em.xml > car.xml
```

# Running the Simulator

You can then run the simulator as follows:

```
python3 -m mujoco.viewer --mjcf=car.xml
```
