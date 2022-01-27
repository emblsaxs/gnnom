"""
Read and parse CRYSOL log files
"""
import os

from gnnom.mysaxsdocument import saxsdocument
from gnnom.utils.log import log_execution_time
import logging
logger = logging.getLogger(__name__)

@log_execution_time(logger)
def parseCrysolLogs(logFiles, par):
    """
    Parse crysol log files for rg,mw,dmax or volume and returns the
    double array of [[params]] and csv file "filename, param\n..."
    """
    parameters = []
    outCsv = []
    for f in logFiles:
        # l    = file#os.path.join(args.logPath,file)
        file = os.path.basename(f)
        # print(f)
        lines = [line.strip() for line in open(f)]
        rgdmaxmwv = []
        # Read 'Molecular Weight: 0.4330E+06':
        if par not in ["rg", "dmax", "mw", "vshell", "vexc"]:
            print(f"Wrong parameter {par}! Please enter rg, dmax or mw")
        for line in lines:
            if par == "rg":
                if "slope" in line:
                    rg = float(line.split()[-1])
                    rgdmaxmwv.append(rg)
                    parameters.append(rgdmaxmwv)
                    outCsv.append(file[:-4] + ', ' + str(round(rg, 3)))
                    break
            if par == "dmax":
                if "diameter" in line:
                    dmax = float(line.split()[-1])
                    rgdmaxmwv.append(dmax)
                    parameters.append(rgdmaxmwv)
                    outCsv.append(file[:-4] + ', ' + str(round(dmax, 3)))
                    break
            if par == "mw":
                if "Weight" in line:
                    mw = float(line.split()[2])
                    # print(f"{file}: {mw} kDa")
                    rgdmaxmwv.append(mw)
                    parameters.append(rgdmaxmwv)
                    outCsv.append(file[:-4] + ', ' + str(round(mw, 3)))
                    break
            if par == "vshell":
                if "Shell    volume" in line:
                    vshell = float(line.split()[3])
                    # print(f"{file}: {v} nm^3")
                    rgdmaxmwv.append(vshell)
                    parameters.append(rgdmaxmwv)
                    outCsv.append(file[:-4] + ', ' + str(round(vshell, 3)))
                    break
            if par == "vexc":
                if "Excluded Volume" in line:
                    vexc = float(line.split()[-1])
                    rgdmaxmwv.append(vexc)
                    parameters.append(rgdmaxmwv)
                    outCsv.append(file[:-4] + ', ' + str(round(vexc, 3)))
                    break

    return parameters, outCsv

@log_execution_time(logger)
def readDatsAndLogs(dataFiles, logPath, firstPointIndex, lastPointIndex):
    """
    Reads *.dat files, finds corresponding crysol log files, returns
    2d list of intensities and a list of log files.
    """
    Is = []
    logFiles = []
    for file in dataFiles:
        name = os.path.basename(file)
        if os.path.isdir(file): continue
        l = os.path.join(logPath, os.path.splitext(name)[0]) + ".log"
        if not os.path.exists(l):
            dataFiles.remove(file)  # This has no effect?..
            print(f"No logs: removed from {file}")
            continue
        cur, prop = saxsdocument.read(file)
        Is.append(cur['I'][firstPointIndex:lastPointIndex])
        logFiles.append(l)
    return Is, logFiles

@log_execution_time(logger)
def readLogs(dataFiles, logPath):
    """
    Returns absolute path to *.log files as a list.
    """
    logFiles = []
    for file in dataFiles:
        name = os.path.basename(file)
        if os.path.isdir(file): continue
        log = name[:-4] + ".log"
        l = os.path.join(logPath, log)
        if os.path.exists(l) == False:
            dataFiles.remove(file)
            print(f"No logs for {file}")
        logFiles.append(l)
    return logFiles
