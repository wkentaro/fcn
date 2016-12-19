import numpy as np

import fcn


class APC2016DatasetBase(fcn.datasets.SegmentationDatasetBase):

    label_names = [
        'background',
        'barkely_hide_bones',
        'cherokee_easy_tee_shirt',
        'clorox_utility_brush',
        'cloud_b_plush_bear',
        'command_hooks',
        'cool_shot_glue_sticks',
        'crayola_24_ct',
        'creativity_chenille_stems',
        'dasani_water_bottle',
        'dove_beauty_bar',
        'dr_browns_bottle_brush',
        'easter_turtle_sippy_cup',
        'elmers_washable_no_run_school_glue',
        'expo_dry_erase_board_eraser',
        'fiskars_scissors_red',
        'fitness_gear_3lb_dumbbell',
        'folgers_classic_roast_coffee',
        'hanes_tube_socks',
        'i_am_a_bunny_book',
        'jane_eyre_dvd',
        'kleenex_paper_towels',
        'kleenex_tissue_box',
        'kyjen_squeakin_eggs_plush_puppies',
        'laugh_out_loud_joke_book',
        'oral_b_toothbrush_green',
        'oral_b_toothbrush_red',
        'peva_shower_curtain_liner',
        'platinum_pets_dog_bowl',
        'rawlings_baseball',
        'rolodex_jumbo_pencil_cup',
        'safety_first_outlet_plugs',
        'scotch_bubble_mailer',
        'scotch_duct_tape',
        'soft_white_lightbulb',
        'staples_index_cards',
        'ticonderoga_12_pencils',
        'up_glucose_bottle',
        'womens_knit_gloves',
        'woods_extension_cord',
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
