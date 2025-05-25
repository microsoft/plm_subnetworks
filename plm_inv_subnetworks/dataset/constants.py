SUBNETWORK_LEVELS_FOR_PLOTTING = {
    'cath_class_code': 'Class',
    'cath_architecture_code': 'Architecture',
    'cath_topology_code': 'Topology',
    'cath_homologous_superfamily_code': 'H. Superfam.',
    'residue': 'Residue',
    'random_seq': 'Random',
}

PRETTY_NAMES = {
    "cath_class_code":"class",
    "cath_architecture_code":"arch",
    "cath_topology_code": "topo",
    "cath_homologous_superfamily_code": "hsuperfam",
    "residue": "res",   
}

DSSP_3 = {
    'H': 'H',  # Alpha-helix
    'G': 'H',  # 3-10 helix
    'I': 'H',  # Pi-helix
    'E': 'E',  # Beta-strand
    'B': 'E',  # Beta-bridge
    'T': 'L',  # Turn (part of loop)
    'S': 'L',  # Bend (part of loop)
    '-': 'L'   # Coil/loop
    
}


DSSP_8_RESIDUES = {'G', 'H', 'T', 'E', 'S', 'L', 'B', "I"}

DSSP_8 = {
        'H': 'H',  # Alpha-helix
        'G': 'G',  # 3-10 helix
        'I': 'I',  # Pi-helix (Now distinct in UniRep)
        'E': 'E',  # Beta-strand
        'B': 'B',  # Beta-bridge
        'T': 'T',  # Turn
        'S': 'S',  # Bend
        '-': 'L'   # Coil/loop
    }


DSSP_3_REVERSE = {
    'H': {'H', 'G', 'I'},  # Helix-related structures
    'E': {'E', 'B'},       # Beta structures
    'L': {'T', 'S', '-', 'C'}   # Loops, turns, and bends
}


HELIX_SET = {'H', 'G', 'I'}
STRAND_SET = {'E', 'B'}
COIL_SET = {'T', 'S', 'L', 'C'}


# SUBNETWORK_LEVELS_FOR_PLOTTING = {
#              "cath_class_code" : "Class",
#              "cath_architecture_code": "Architecture",
#              "cath_topology_code": "Topology",
#              "cath_homologous_superfamily_code": "H. Superfam.",
#              "residue": "Residue",
#              }

# CATH_LEVEL_TO_NAME = {"class" : 1,
#              "architecture": 2,
#              "topology": 3,
#              "homologous_superfamily": 4,
#              "domain_num": 5,
#              }

# CATH_NAME_TO_LEVEL = {1: "class",
#                  2: "architecture",
#                  3: "topology",
#                  4: "homologous_superfamily",
#                  5: "domain_num"
#                  }


# CATH_CLASSES = {
#     1: 'Mainly Alpha',
#     2: 'Mainly Beta', 
#     3: 'Alpha Beta',
#     4: 'Few Secondary Structures',
#     6: 'Special'
# }


# CATH_CLASSES_TRAINING = {
#     'alpha': 1,
#     'beta': 2, 
#     'alphabeta': 3,
#     'fewss': 4,
#     'special': 6
# }

# TOP5_CLASSES = [1, 2, 3] #, 4, 6]

# TOP5_ARCHITECTURE = [40, 10, 30, 20, 60]

# TOP5_TOPOLOGY = [50, 40, 10, 20, 120]

# TOP5_HOMOLOGOUS_SUPERFAMILY = [10, 20, 30, 40, 60]

# TOP5_DOMAIN = [0, 1, 2, 3, 4]


# TOP10_ARCHITECTURE = [40, 10, 30, 20, 60, 90, 25, 50, 70, 80]
# TOP10_TOPOLOGY = [50, 40, 10, 20, 120, 70, 30, 58, 190, 450, 60]
# TOP10_HOMOLOGOUS_SUPERFAMILY = [10, 20, 30, 40, 60, 80, 300, 50, 70, 150]

# ARCH_25_TOPOLOGIES = [10, 20, 40, 50, 60, 70]

# TOP5_CLASS_CODES = ['1', '2', '3']#['3', '1', '2'] #, '6', '4']
# TOP10_ARCHITECTURE_CODES = ['3.40', '3.30', '1.10', '2.60', '1.20', '3.90', '2.40', '3.10'] #, "3.60", "2.30", "1.25"] #, '3.20', '2.30']
# TOP10_TOPOLOGY_CODES = ['3.40.50', '2.60.40', '3.30.70', '2.60.120', '3.20.20', '1.10.10', '1.25.40', '2.40.50', '1.20.58', '1.20.120']
# TOP10_HOMOLOGOUS_SUPERFAMILY_CODES = ['3.40.50.300', '2.60.40.10', '3.40.50.720', '1.10.10.10', '3.20.20.80', '3.40.190.10', '3.40.50.150', '3.40.50.1820', '2.40.50.140', '3.40.30.10']
# TOP10_DOMAIN_CODES = ['3.40.50.720.1', '3.40.50.300.0', '3.40.50.1820.0', '3.40.50.300.1', '2.60.40.10.2', '1.10.10.10.0', '2.60.40.10.1', '3.20.20.70.0', '3.40.190.10.1', '3.40.190.10.2']
