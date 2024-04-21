# sequence list for single-view human&cloth reconstruction and image-based human parsing
SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING = {

    '00122': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take8'],
        'Outer': ['Take11', 'Take16'],
    },

    '00123': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take10', 'Take11'],
    },

    '00127': {
        'Gender': 'male',
        'Inner': ['Take8', 'Take9'],
        'Outer': ['Take16', 'Take18'],
    },

    '00129': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take11', 'Take13'],
    },

    '00134': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take6'],
        'Outer': ['Take12', 'Take19'],
    },

    '00135': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take21', 'Take24'],
    },

    '00136': {
        'Gender': 'female',
        'Inner': ['Take8', 'Take12'],
        'Outer': ['Take19', 'Take28'],
    },

    '00137': {
        'Gender': 'female',
        'Inner': ['Take5', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00140': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take8'],
        'Outer': ['Take19', 'Take21'],
    },

    '00147': {
        'Gender': 'female',
        'Inner': ['Take11', 'Take12'],
        'Outer': ['Take16', 'Take19'],
    },

    '00148': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00149': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take12'],
        'Outer': ['Take14', 'Take24'],
    },

    '00151': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take15', 'Take20'],
    },

    '00152': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00154': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take9'],
        'Outer': ['Take20', 'Take21'],
    },

    '00156': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take14', 'Take19'],
    },

    '00160': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take17', 'Take18'],
    },

    '00163': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take13', 'Take15'],
    },

    '00167': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take9'],
        'Outer': ['Take12', 'Take14'],
    },

    '00168': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take11', 'Take16'],
    },

    '00169': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take10'],
        'Outer': ['Take17', 'Take19'],
    },

    '00170': {
        'Gender': 'female',
        'Inner': ['Take9', 'Take11'],
        'Outer': ['Take15', 'Take24'],
    },

    '00174': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take9'],
        'Outer': ['Take13', 'Take15'],
    },

    '00175': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take13', 'Take20'],
    },

    '00176': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take11', 'Take14'],
    },

    '00179': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take13', 'Take15'],
    },

    '00180': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00185': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00187': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take6'],
        'Outer': ['Take10', 'Take15'],
    },

    '00188': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take12', 'Take18'],
    },

    '00190': {
        'Gender': 'female',
        'Inner': ['Take2', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00191': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take13', 'Take19'],
    },

}

# sequence list for video-based human reconstruction and human representation learning
SUBJ_OUTFIT_SEQ_HUMANRECON_HUMANAVATAR = {
    'Inner': {
        '00148': {'Train': ['Take1', 'Take2', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9', 'Take10'], 'Test': ['Take7']},
        '00152': {'Train': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take9'], 'Test': ['Take8']},
        '00154': {'Train': ['Take1', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take11'], 'Test': ['Take9']},
        '00185': {'Train': ['Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9'], 'Test': ['Take7']},
    },
    'Outer': {
        '00127': {'Train': ['Take11', 'Take13', 'Take14', 'Take15', 'Take16', 'Take17', 'Take19'], 'Test': ['Take18']},
        '00137': {'Train': ['Take12', 'Take13', 'Take14', 'Take15', 'Take17', 'Take18', 'Take19', 'Take20', 'Take21'], 'Test': ['Take16']},
        '00149': {'Train': ['Take14', 'Take15', 'Take16', 'Take17', 'Take20', 'Take22', 'Take24', 'Take25'], 'Test': ['Take21']},
        '00188': {'Train': ['Take10', 'Take11', 'Take12', 'Take15', 'Take16', 'Take17', 'Take18'], 'Test': ['Take14']},
    }
}

# sequence list for clothing simulation
SUBJ_OUTFIT_SEQ_CLOTHSIMULATION = {
    'lower': {
        '00129': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take22'],
        '00156': ['Take2', 'Take3', 'Take4', 'Take7', 'Take8', 'Take9'],
        '00152': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
        '00174': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
    },
    'upper': {
        '00127': ['Take5', 'Take6', 'Take7', 'Take8', 'Take9', 'Take10'],
        '00140': ['Take1', 'Take3', 'Take4', 'Take6', 'Take7', 'Take8'],
        '00147': ['Take1', 'Take2', 'Take3', 'Take4', 'Take6', 'Take9'],
        '00180': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
    },
    'dress': {
        '00185': ['Take1', 'Take2', 'Take3', 'Take4', 'Take7', 'Take8'],
        '00148': ['Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take9'],
        '00170': ['Take1', 'Take3', 'Take5', 'Take7', 'Take8', 'Take9'],
        '00187': ['Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6'],
    },
    'outer': {
        '00123': ['Take8', 'Take9', 'Take10', 'Take11', 'Take12', 'Take13'],
        '00152': ['Take10', 'Take12', 'Take15', 'Take17', 'Take18', 'Take19'],
        '00176': ['Take9', 'Take10', 'Take11', 'Take12', 'Take13', 'Take14'],
        '00190': ['Take10', 'Take11', 'Take13', 'Take14', 'Take15', 'Take16'],
    },   
}
