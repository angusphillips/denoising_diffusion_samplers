param_settings = {
    "difficult_gaussian": {
        "pis": {
            "sigma": {  # Not tuned as otherwise would be exact
                2: 1.0,
                4: 1.0,
                8: 1.0,
                16: 1.0,
            }
        },
        "dds": {
            "alpha": {
                1: 0.86,
                2: 0.86,
                4: 1.00,
                8: 0.96,
                16: 0.82,
            }
        },
    },
    "difficult_2d": {
        "pis": {
            "sigma": {
                2: 2.40,
                4: 2.20,
                8: 1.92,
                16: 1.88,
            }
        },
        "dds": {
            "alpha": {
                1: 0.28,
                2: 0.28,
                4: 0.36,
                8: 0.52,
                16: 0.54,
            }
        },
    },
    "brownian": {
        "pis": {
            "sigma": {
                2: 0.08,
                4: 0.10,
                8: 0.10,
                16: 0.13,
            }
        },
        "dds": {
            "alpha": {
                1: 0.76,
                2: 0.76,
                4: 0.84,
                8: 0.80,
                16: 0.72,
            }
        },
    },
    "funnel": {
        "pis": {
            "sigma": {
                1: 1.5,
                2: 1.0,
                4: 1.0,
                8: 1.0,
                16: 1.0,
            }
        },
        "dds": {
            "alpha": {
                1: 0.60,
                2: 0.68,
                4: 0.68,
                8: 0.60,
                16: 0.64,
            }
        },
    },
    "ion": {
        "pis": {
            "sigma": {
                1: 0.37,
                2: 0.40,
                4: 0.40,
                8: 0.46,
                16: 0.46,
            }
        },
        "dds": {
            "alpha": {
                1: 0.68,
                2: 0.80,
                4: 0.74,
                8: 0.64,
                16: 0.52,
            }
        },
    },
    "sonar": {
        "pis": {
            "sigma": {
                1: 0.25,
                2: 0.31,
                4: 0.40,
                8: 0.46,
                16: 0.49,
            }
        },
        "dds": {
            "alpha": {
                1: 0.68,
                2: 0.82,
                4: 0.78,
                8: 0.64,
                16: 0.50,
            }
        },
    },
    "lgcp": {
        "pis": {
            "sigma": {
                1: 1.36,
                2: 1.64,
                4: 1.78,
                8: 1.99,
                16: 2.06,
            }
        },
        "dds": {
            "alpha": {
                1: 0.74,
                2: 0.62,
                4: 0.60,
                8: 0.44,
                16: 0.26,
            }
        },
    },
    "gmm1": {
        "pis": {
            "sigma": {
                1: 15.0,
                2: 14.0,
                4: 15.0,
                8: 15.0,
                16: 16.0,
            }
        },
        "dds": {
            "alpha": {
                1: 0.22,
                2: 0.28,
                4: 0.24,
                8: 0.20,
                16: 0.22,
            }
        },
    },
    "gmm2": {
        "pis": {
            "sigma": {
                1: 4.4,
                2: 1.3,
                4: 1.3,
                8: 7.3,
                16: 8.7,
            }
        },
        "dds": {
            "alpha": {
                1: 0.14,
                2: 0.18,
                4: 0.18,
                8: 0.16,
                16: 0.16,
            }
        },
    },
    "gmm5": {
        "pis": {
            "sigma": {
                1: 1.3,
                2: 1.3,
                4: 1.4,
                8: 1.4,
                16: 1.4,
            }
        },
        "dds": {
            "alpha": {
                1: 0.20,
                2: 0.28,
                4: 0.26,
                8: 0.24,
                16: 0.20,
            }
        },
    },
    "gmm10": {
        "pis": {
            "sigma": {
                1: 1.3,
                2: 1.3,
                4: 1.3,
                8: 1.3,
                16: 1.3,
            }
        },
        "dds": {
            "alpha": {
                1: 0.36,
                2: 0.34,
                4: 0.30,
                8: 0.26,
                16: 0.22,
            }
        },
    },
    "gmm20": {
        "pis": {
            "sigma": {
                1: 1.3,
                2: 1.3,
                4: 1.3,
                8: 1.2,
                16: 1.2,
            }
        },
        "dds": {
            "alpha": {
                1: 0.40,
                2: 0.32,
                4: 0.32,
                8: 0.26,
                16: 0.18,
            }
        },
    },
}