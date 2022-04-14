from collections import defaultdict

import numpy as np


class document:
    def __init__(self):
        self.property = defaultdict(list)
        # 0 - s, 1 - I, 2 - Err, 3 - Fit
        self.curve = [[] for i in range(4)]

    def set(self, curve, prop={}):
        self.property = prop
        self.curve = curve


def read(fileName):
    doc = document()
    try:
        with open(fileName) as f:
            for line in f:
                try:
                    # FIXME: if more than one colon in the line
                    if ':' in line:
                        l = line.split(':')
                        key = l[0]
                        val = (':'.join(l[1:]))
                        # key, val = line.split(':')[0:2]
                        doc.property[key].append(val)
                    else:
                        line = line.strip()
                        cols = line.split()
                        cols = list(filter(lambda a: a != '', cols))
                        if len(cols) == 0: continue
                        if len(cols) >= 2 and cols[0][0].isdigit() and cols[0][2].isdigit():
                            # we are in a number section
                            s = float(cols[0])
                            I = float(cols[1])
                            doc.curve[0].append(s)
                            doc.curve[1].append(I)
                            # if dat file with errors
                            if len(cols) == 3:
                                Err = float(cols[2])
                                doc.curve[2].append(Err)
                            # if fit file with errors
                            if len(cols) == 4:
                                Err = float(cols[2])
                                doc.curve[2].append(Err)
                                Fit = float(cols[3])
                                doc.curve[3].append(Fit)
                            # The first line is a title. Five columns contain: (1) experimental scattering vector in inverse angstroms, (2) theoretical intensity in solution, (3) in vacuo, (4) the solvent scattering and (5) the border layer scattering.
                            if len(cols) == 5:
                                vacuum = float(cols[2])
                                doc.curve[2].append(vacuum)
                                solvent = float(cols[3])
                                doc.curve[3].append(solvent)
                                border = float(cols[4])
                                doc.curve[4].append(border)
                except Exception as e:
                    print(f"Warning: for file {fileName}: {e}")
                    pass
        doc.curve = [x for x in doc.curve if x != []]
        return doc
    except Exception as e:
        print(f"Cannot open file {fileName}: {e}")
        pass


def write(path, curve, prop={}, nm=True):
    '''Writes a SAXS curve, takes: (1) path where to write a file, can not create a folder
    (2) curve as list of lists [[s][I][Err}[Fit] and properties in arbitrarory form as
    the python dictionary'''
    head = ""
    foot = ""
    headerKeys = ["SASBDB code"]
    s = np.array(curve[0]).astype(np.float64)
    if nm == False: s = s * 10
    I = np.array(curve[1]).astype(np.float64)
    # split properties on those who go to the header (headerKeys) and the rest that go to the footer
    if prop:
        for hk in headerKeys:
            if hk in prop: head += f"{hk} : {prop[hk]}"
        for key, val in prop.items():
            if key not in headerKeys: foot += f"{key} : {val}\n"
    if len(curve) == 2:
        out = np.vstack((s, I))
    elif len(curve) == 3:
        Err = np.array(curve[2]).astype(np.float64)
        out = np.vstack((s, I, Err))
    elif len(curve) == 4:
        Err = np.array(curve[2]).astype(np.float64)
        Fit = np.array(curve[3]).astype(np.float64)
        out = np.vstack((s, I, Err, Fit))

    np.savetxt(path, np.transpose(out), fmt="%.8e", header=head, footer=foot, comments='')

    # except Exception as e:
    #    print(f"Could not write file {path}: {e}")
