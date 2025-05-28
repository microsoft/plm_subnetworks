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

