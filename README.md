# Vision system for Marble Madness

Yes, I know I'm the only one who bother to **waste my own time writing document** for the code, and the only one who will bother reading it.

## How to install the dependencies

```
pip install -r requirements.txt
```

## How to run the test program

First run the box2d simulator, then run

```
python im2env.py
```

## What is going on

- The program extract the block info based on the image. Currently the image is hard-coded to be the first image in `test_imgs`
    - The important information is the number of blocks, the height and width of the blocks, and the posistion of the destination
    - The starting point of the ball is hard-coded like this: 
    ```python
    ball = {'radius': 0.01,
            'location': [x_bound/3, y_bound],
            'linear_velocity': [0.1, -0.05]
        }
    ```
- The program then sends the environment information through AJAX call to the optimizer, and it will receive the optimized pose of each block. Refer to the document of the optimizer for details