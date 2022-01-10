import numpy as np


def read(fileName):
    properties = {}
    curve = {"s": [], "I": [], "Err": [], "Fit": []}
    try:
        with open(fileName) as f:
            for line in f:
                try:
                    if ':' in line:
                        key, val = line.split(':')[0:2]
                        properties[key] = val
                    else:
                        line = line.strip()
                        cols = line.split()
                        cols = list(filter(lambda a: a != '', cols))
                        if len(cols) == 0: continue
                        if len(cols) >= 2 and cols[0][0].isdigit() and cols[0][2].isdigit():
                            # we are in a number section
                            s = float(cols[0])
                            I = float(cols[1])
                            curve["s"].append(s)
                            curve["I"].append(I)
                            # if dat file with errors
                            if len(cols) == 3:
                                Err = float(cols[2])
                                curve["Err"].append(Err)
                            # if fit file with errors
                            if len(cols) == 4:
                                Err = float(cols[2])
                                curve["Err"].append(Err)
                                Fit = float(cols[3])
                                curve["Fit"].append(Fit)
                except Exception as e:
                    print(f"Warning: for file {fileName}: {e}")
                    pass
        return curve, properties
    except Exception as e:
        print(f"Cannot open file {fileName}: {e}")
        pass


def write(path, curve, prop=None):
    head = ""
    foot = ""
    headerKeys = ["Sample", "Sample description", "parent"]
    try:
        s = np.array(curve["s"]).astype(np.float64)
        I = np.array(curve["I"]).astype(np.float64)
        if prop:
            for hk in headerKeys:
                if hk in prop:
                    st = prop[hk]
                    head += f"{hk}: {st}\n"
            for key, val in prop.items():
                # if key not in headerKeys:
                foot += f"{key}: {val}\n"
        if "Err" in curve.keys() and "Fit" not in curve.keys():
            Err = np.array(curve["Err"]).astype(np.float64)
            out = np.vstack((s, I, Err))
        elif "Err" in curve.keys() and "Fit" in curve.keys():
            Err = np.array(curve["Err"]).astype(np.float64)
            Fit = np.array(curve["Fit"]).astype(np.float64)
            out = np.vstack((s, I, Err, Fit))
        elif "Err" not in curve.keys() and "Fit" not in curve.keys():
            out = np.vstack((s, I))
        np.savetxt(path, np.transpose(out), fmt="%.6e", header=head, footer=foot, comments='')

    except Exception as e:
        print(f"Could not write file {path}: {e}")
