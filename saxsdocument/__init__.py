import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read(fileName):
    properties = {}
    curve = {"s": [], "I": [], "Err": [], "Fit" : []}
    try:
        with open(fileName) as f:
            for line in f:
                try:
                    line = line.strip()
                    cols = line.split(" ")
                    cols = list(filter(lambda a: a != '', cols))
                    if is_number(cols[0]) and len(cols) >= 2:
                        curve["s"].append(cols[0])
                        curve["I"].append(cols[1])
                        # if dat file with errors
                        if len(cols) == 3:
                            curve["Err"].append(cols[2])
                        # if fit file with errors
                        if len(cols) == 4:
                            curve["Err"].append(cols[2])
                            curve["Fit"].append(cols[3])
                    elif ":" in line:
                        key, val = line.split(":")[0:2]
                        properties[key] = val
                except Exception as e:
                    print(e)
                    pass
        return properties, curve
    except Exception as e:
        print(f"Cannot open file {fileName}: {e}")
        pass


def write(path, curve, prop=None):
    head = ""
    foot = ""
    headerKeys = ["Sample", "Sample description"]
    try:
        s = np.array(curve["s"]).astype(np.float64)
        I = np.array(curve["I"]).astype(np.float64)
        if prop:
            for hk in headerKeys:
                if hk in prop:
                    st = prop[hk]
                    head += f"{hk} : {st}\n"
            for key, val in prop.items():
                if key not in headerKeys:
                   foot += f"{key} : {val}\n"
        if len(curve["Err"]) > 0 and len(curve["Fit"]) == 0:
            Err = np.array(curve["Err"]).astype(np.float64)
            file = np.vstack((s, I, Err))
        if len(curve["Err"]) > 0 and len(curve["Fit"]) > 0:
            Err = np.array(curve["Err"]).astype(np.float64)
            Fit = np.array(curve["Fit"]).astype(np.float64)
            file = np.vstack((s, I, Err, Fit))
        else:
            file = np.vstack((s, I))
        np.savetxt(path, np.transpose(file), fmt="%.8e", header=head, footer=foot)

    except Exception as e:
        print(f"Could not write file {path}: {e}")
