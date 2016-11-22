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

## 0.0 Usage 

The TL;DR is:

To set up the system, ask Sam. This involves
data scraping and editing a config file. Then
the following commands should work when one
is in working directory `stronglenses`:

    python -m engine.visualize     /* display samples from the visualization set   */
    python -m engine.train_nn 20   /* train a net for 20 epochs (on the train set) */
    python -m engine.train_nn 0    /* test a net (on the test set)                 */
    python -m engine.view_history  /* inspect training history                     */
    python -m engine.view_curves   /* generate yield and confidence curves         */

The latter two commands save to subdirectory `discussion/figures`. 

## 0.1. Goals

For this project, we must:

                                                        PROGRESS        TIME GOAL       TIME ACTUAL
                                                                        (hours)         (hours)
    Project STRONG-LENS                                 [=>-]           ???             ...            
  
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

## 0.2. Directory Structure

Our project directory structure, simplified of clutter, is
as follows. Asterisks denote skeletal files. Our data, model
checkpoints, and training logs we keep elsewhere, as specified
in `config.json`. Each worker on this project will likely
have a different directory structure external to the following,
so we leave as adjustable the paths in `config.json`. See
`config.json.example` for a config file that works for Sam.

    stronglenses/
        README.md
        config.json
        discussion/
            notes/
                coding_standards.md (*)
                ...
            figures/
                ...
        engine/
            gui_demo.py (*)
            train_nn.py
            view_curves.py
            view_history.py
            visualize.py
        data_scrape/
            fetch_data.py
            prepare_data.py
        model/
            fetch_model.py
            make_model.py
            train.py
            test.py
        utils/
            config.py 
            terminal.py
            file_io.py (*)

# 1. System Design

To be filled in...

