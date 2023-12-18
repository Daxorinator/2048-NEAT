from selenium import webdriver
import os.path
import json, time, neat
from selenium.webdriver.common.by import By


class Game:
    site_path = os.path.abspath("./2048.html")
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    browser = webdriver.Firefox(options=options)
    
    def __init__(self):
        self.browser.get(f'file://{self.site_path}')
        time.sleep(1)
        self.game_state = json.loads(self.browser.execute_script("return localStorage.getItem('gameState')"))

    def reset(self):
        reset_button = self.browser.find_element(By.CLASS_NAME, "restart-button")
        reset_button.click()

    def update_state(self):
        self.game_state = json.loads(self.browser.execute_script("return localStorage.getItem('gameState')"))

    def send_input(self, key):
        input_body = self.browser.find_element(By.TAG_NAME, "body")
        input_body.send_keys(key)

class NeuralNetwork:
    def __init__(self):
        # Load NEAT Configuration
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, './NEAT.config')
        
        # Create the population (top level object for a NEAT run)
        self.pop = neat.Population(self.config)

        # Add some statistics reporting via STDOUT
        self.pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.pop.add_reporter(stats)
        self.pop.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations
        self.winner = self.pop.run(self.evaluate, 300)

        # Print the winning genome
        print(f'Best genome:\n{self.winner}')

    def evaluate(self):
        pass

if __name__ == '__main__':
    nn = NeuralNetwork()

# for col in json_state['grid']['cells']:
#     for cell in col:
#         if cell:
#             print(cell['value'])