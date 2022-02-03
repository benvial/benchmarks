#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT



import psutil
import platform
import cpuinfo
import GPUtil

# Let's make a function that converts a large number of bytes into a scaled format (e.g in kilo, mega, Giga, etc.):

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor




def get_info(verbose=False):
    # System Information
    # 
    info = dict()
    uname = platform.uname()
    cpu_info = cpuinfo.get_cpu_info()
    
    info["uname"] = uname
    info["cpu_info"] = cpu_info

        
    # CPU Information
    # 
    # Let's get some CPU information, such as the total number of cores, usage, etc:

    # number of cores
    cores = dict(physical=psutil.cpu_count(logical=False),total=psutil.cpu_count(logical=True))
    
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    cores["cpufreq"]=cpufreq
    # CPU usage
    percentages=[]
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=0.1)):
        percentages.append(percentage)
    
    cores["individual usage"] = percentages
    cores["total usage"] = psutil.cpu_percent()
    
    info["cores"] = cores

    # psutil's cpu_count() function returns number of cores, whereas cpu_freq() function returns CPU frequency as a namedtuple including current, min, and max frequency expressed in Mhz, you can set percpu=True to get per CPU frequency.
    # 
    # cpu_percent() method returns a float representing the current CPU utilization as a percentage, setting interval to 1 (seconds) will compare system CPU times elapsed before and after a second, we set percpu to True in order to get CPU usage of each core.
    # Memory Usage

    # Memory Information
    # get the memory details
    svmem = psutil.virtual_memory()
    info["svmem"] = svmem
    
    
    try:
        gpus = GPUtil.getGPUs()
    except:
        gpus = []
        
    info["gpu"] = gpus

    
    
    if verbose:
        

        print("="*33, "System Information", "="*33)
        print(f"System: {uname.system}")
        print(f"Node Name: {uname.node}")
        print(f"Release: {uname.release}")
        print(f"Version: {uname.version}")
        print(f"Machine: {uname.machine}")
        print(f"Processor: {cpu_info['brand_raw']}")
        print(f"Python: {cpu_info['python_version']}")

        # let's print CPU information
        print("="*33, "CPU Info", "="*33)
        print("Physical cores:",cores["physical"])
        print("Total cores:",cores["total"])
        print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
        print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
        print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
        
        print("CPU Usage Per Core:")
        for i, percentage in enumerate(percentages):
            print(f"Core {i}: {percentage}%")
        print(f"Total CPU Usage: {cores['total usage']}%")
        
        print("="*33, "Memory Information", "="*33)
        print(f"Total: {get_size(svmem.total)}")
        print(f"Available: {get_size(svmem.available)}")
        print(f"Used: {get_size(svmem.used)}")
        print(f"Percentage: {svmem.percent}%")
        

        print("="*33, "GPU Information", "="*33)

        for gpu in gpus:

            print(f"ID: {gpu.id}")
            print(f"Name: {gpu.name}")
            print(f"Load: {gpu.load*100}%")
            print(f"free memory: {gpu.memoryFree}MB")
            print(f"used memory: {gpu.memoryUsed}MB")
            print(f"total memory: {gpu.memoryTotal}MB")
            print(f"temperature: {gpu.temperature} °C")
            print(f"uuid: {gpu.uuid}")

    return info
