![an image of the simulator](images/ft_grandprix_volume_2.png)


# Windows Setup

## Automatic Install

Download and double click the installer from [here](https://github.com/FT-Autonomous/ft_grandprix/releases/download/v0.0.0-alpha/run_ft_grandprix.bat).

## Manual

[![Windows 10 / Windows 11](http://img.youtube.com/vi/-0va_8FHMhQ/0.jpg)](http://www.youtube.com/watch?v=-0va_8FHMhQ)

Make sure python3 is installed. You can get it from the windows store.

Install microsoft visual c++ redistributable for your architecture from [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

Failing to install this will result in an error complaining that "mujoco.dll" cannot be found, even though it exists.
An explanation for this is provided in the ctypes module page.

"On Windows creating a CDLL instance may fail even if the DLL name exists. When a dependent DLL of the loaded DLL is not found, a OSError error is raised with the message “[WinError 126] The specified module could not be found”.
This error message does not contain the name of the missing DLL because the Windows API does not return this information making this error hard to diagnose.
To resolve this error and determine which DLL is not found, you need to find the list of dependent DLLs and determine which one is not found using Windows debugging and tracing tools."

By default, windows blocks running other powershell scripts, which is needed for installing python packages in a virtual environment.
Open another powershell window as administrator briefly.
Enter the following ([explanation](https://stackoverflow.com/questions/1365081/virtualenv-in-powershell)).

```
set-executionpolicy RemoteSigned
```

You may then close the powershell instance running as administrator.

Go to the [the github page](https://github.com/FT-Autonomous/ft_grandprix) and click *download zip*.
Extract the `ft_grandprix-main` folder somewhere.

Open the folder where you extracted the ZIP, shift-right-click and
then select "open powershell". Enter the following:

```
python3 -m venv env
.\env\Scripts\activate.ps1
pip3 install -r requirements.txt
python3 -m ft_grandprix.custom
```

# MacOS / Linux Setup

[![MacOS](http://img.youtube.com/vi/jvQxtafER_U/0.jpg)](http://www.youtube.com/watch?v=jvQxtafER_U)
[![Linux / WSL](http://img.youtube.com/vi/RV46Vvxnug8/0.jpg)](http://www.youtube.com/watch?v=RV46Vvxnug8)

```
git clone https://github.com/FT-Autonomous/ft_grandprix
git pull
python3 -m venv env
. env/bin/activate
cd ft_grandprix
pip3 install -r requirements.txt
python3 -m ft_grandprix.custom
```

# Getting Updates

Install `git`. To update your code, type in:

```
git pull
```

# Writing a Driver

The drivers "ft_grandprix.nidc" and "ft_grandprix.fast" are bundled with the project the only relevant function is the `process_lidar` function, which generates a speed and a steering angle.
To start writing a driver, you can start by modifying the `drivers.template` drive.

# Reporting Bugs

Make an issue on github and I will try to fix it.
