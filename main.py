from selenium import webdriver
import os.path
import json, time, neat, random
from selenium.webdriver.common.by import By


class Game:
    """
    Represents a game of 2048 running in a headless Firefox instance using Selenium WebDriver
    The NEAT neural networks play games represented by this class.
    """

    # Get the absolute path of the 2048 HTML Page
    site_path = os.path.abspath("./2048.html")
    
    def __init__(self):
        # Configure WebDriver to run Firefox in headless mode
        self.service = webdriver.firefox.service.Service(port=random.randint(4444, 9999))
        self.options = webdriver.FirefoxOptions()
        self.options.add_argument("--headless")

        # Instantiate a new browser to play the game
        self.browser = webdriver.Firefox(service=self.service, options=self.options)
        
        # By default the game is not lost
        self.is_over = False
        self.epoch = time.time()
        self.timer = 0
        self.score = 0

        # GET request for the 2048 page
        self.browser.get(f'file://{self.site_path}')
        # Sleep for a second to let the game boot up
        time.sleep(0.1)
        # Grab the initial state of the board
        self.update_state()

    def reset(self):
        # Find the reset button and click it
        reset_button = self.browser.find_element(By.CLASS_NAME, "restart-button")
        reset_button.click()

    def update_state(self):
        # Grab the game state from local storage and decode its JSON into a dictionary
        self.game_state = json.loads(self.browser.execute_script("return localStorage.getItem('gameState')"))

        # If the Game Over flag is set, the state should reflect that so the evaluation can be stopped
        if self.game_state["over"] == 'true':
            self.is_over = True

        self.score = self.game_state["score"]

        # Define a list to store the 16 grid values
        # Pull each cell from the grid into the list, replacing Null's with 0's
        self.grid = []
        for col in self.game_state['grid']['cells']:
            for cell in col:
                self.grid.append(cell['value'] if cell else 0)

    def send_input(self, key):
        input_body = self.browser.find_element(By.TAG_NAME, "body")
        input_body.send_keys(key)

    def quit(self):
        self.browser.close()

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

        # Run for up to 300 generations
        self.winner = self.pop.run(self.evaluate, 300)

        # Print the winning genome
        print(f'Best genome:\n{self.winner}')

    def evaluate(self, genomes_in, config):
        genomes = []
        neural_nets = []
        games = []

        for _, genome in genomes_in:
            genome.fitness = 0
            games.append(Game())
            genomes.append(genome)
            neural_nets.append(
                neat.nn.FeedForwardNetwork.create(genome, config)
            )


        while True:
            if len(games) == 0:
                break

            for idx, game in enumerate(games):
                game.update_state()
                if not (game.is_over or game.timer > 3):
                    nn_output = neural_nets[idx].activate(game.grid)
                    key = nn_output.index(max(nn_output))
                    match key:
                        case 0:
                            game.send_input('w')
                        case 1:
                            game.send_input('a')
                        case 2:
                            game.send_input('s')
                        case 3:
                            game.send_input('d')
                else:
                    game.quit()
                    games.pop(idx)
                    genomes.pop(idx)
                    neural_nets.pop(idx)

            for idx, game in enumerate(games):
                game.update_state()
                if game.score > genomes[idx].fitness:
                    genomes[idx].fitness = game.score
                else:
                    game.timer = time.time() - game.epoch


if __name__ == '__main__':
    nn = NeuralNetwork()
