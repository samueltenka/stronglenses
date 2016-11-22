# stronglenses
    author: sam tenka

    date: 2016-10-12

    descr: Strong Lens Detection via Neural Net.

           We'll classify deepfield astronomical
           images by whether or not they contain
           a strong gravitational lens. This project
           is part of the Michigan Data Science
           Team's 2016 activities.

# 0. Project Outline

## 0.0. Goals

For this project, we must:

                                                        PROGRESS        TIME GOAL       TIME ACTUAL
                                                                        (hours)         (hours)
    Project STRONG-LENS                                 [---]           ???             ...            
  
    0. Gather data.                                     [===]           2.5             4.0 
        0.0. Download labled images.                            [===]           1.0            1.0
        0.1. Create and populate data directory.                [===]           0.5            0.5
        0.2. Preprocess into numpy arrays.                      [===]           0.5            2.0
        0.3. Split into train/test/val.                         [===]           0.5            0.5
    1. Baseline System.                                 [===]           6.0             9.0
        1.0. Develop architectures.                             [===]           1.5            3.0
        1.1. Train.                                             [===]           1.5            2.0
        1.2. Test.                                              [===]           0.5            0.5
        1.3. Visualize outputs.                                 [===]           1.5            2.5
        1.4. Compose report.                                    [===]           1.0            1.0
    2. Iterate                                          [---]           ???             ...
        2.0. Discuss with BNord et al                           [---]           0.5             ...
        2.1. Act on discussion                                  [---]           ???             ...

## 0.1. Directory Structure

Our project directory structure will be as follows. Most importantly,
`stronglenses/README.md` documents our project, while
`stronglenses/engine/gui_demo.py` implements a graphical demonstration.
Run the latter (from `news_seg/`) with `python -m engine.gui_demo` 
to interactively segment images. 

    data/
        train/
            ...
        test/
            ...
        validate/
            ...
    news_seg/
        README.md
        __init__.py
        data_scrape/
            __init__.py
        engine/
            __init__.py
            engine.config
            gui_demo.py
        model/
            __init__.py
            model.py
            train.py
            test.py
        utils/
            terminal.py
            file_io.py

# 1. System Design

# 2. Demonstration


