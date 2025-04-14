import numpy as np

class RRT:
    class Node:
        def __init__(self,pos):
            self.pos = (float(pos[0]), float(pos[1]))
            self.parent = []
    
    def __init__(
        self,
        start,
        goal,
        obstacles,
        x_bound,
        y_bound,
        max = 5000,
        resolution = 20.0, #0.2,
        goal_dist = 20.0,#0.2,
        map_resolution = None,
        origin_x = 0,
        origin_y = 0,
        map_to_pixel = None
    ):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.obstacles = obstacles
        self.x_bound = (0, obstacles.shape[1])  
        self.y_bound = (0, obstacles.shape[0])
        self.x_width = x_bound[1] - x_bound[0]
        self.y_height = y_bound[1] - y_bound[0]
        self.max = max
        self.resolution = resolution
        self.goal_dist = goal_dist
        self.node_list = []
        self.map_resolution = map_resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.map_to_pixel = map_to_pixel

    def plan(self):
        print("planning")
        self.node_list = [self.start]
        for i in range(self.max):
            random = self.get_random()
            nearest = self.get_nearest(self.node_list, random)
            new = self.steer(nearest, random)

            if not self.collision(nearest, new):
                print(f"no collision, new:{new.pos}")
                self.node_list.append(new)
            # else:
                # print("collided")

            if self.is_goal_reached(new):
                # final_node = self.steer(###)
                # if not self.collision(###):
                #     return self.create_path(len(self.node_list)-1)
                print("goal reached")
                return self.create_path(len(self.node_list)-1)

        return None

    def steer(self, nearest, random):
        theta = np.arctan2(random.pos[1] - nearest.pos[1], random.pos[0] - nearest.pos[0])
        new_pos = (nearest.pos[0] + self.resolution * np.cos(theta), nearest.pos[1] + self.resolution * np.sin(theta))
        new_node = self.Node(new_pos)
        new_node.parent = nearest

        return new_node

    def get_random(self):
        if np.random.rand() < 0.1:
            return self.goal
        else:
            x_rand = np.random.uniform(self.x_bound[0], self.x_bound[1])
            y_rand = np.random.uniform(self.y_bound[0], self.y_bound[1])
            return self.Node((x_rand, y_rand))
        # x_rand = np.random.uniform(self.x_bound[0], self.x_bound[1])
        # y_rand = np.random.uniform(self.y_bound[0], self.y_bound[1])
        # return self.Node((x_rand, y_rand))
        
        # rand = np.random.randint(0, self.x_width * self.y_height)
        # x_rand = float(rand % self.x_width)
        # y_rand = float(rand // self.y_height)
        # rand_node = self.Node((x_rand,y_rand))

        # return rand_node

    def get_nearest(self, node_list, random):
        dist = [np.linalg.norm(np.array(node.pos) - np.array(random.pos)) for node in node_list]
        nearest_index = np.argmin(dist)

        return self.node_list[nearest_index]

    def collision(self, nearest, new):
        x0, y0 = nearest.pos
        x1, y1 = new.pos
        dist = np.linalg.norm([x1 - x0, y1 - y0])
        steps = max(1, int(dist / self.resolution))  # 👈 fix here
        # steps = int(dist / self.resolution)

        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            px = int(np.round(x))
            py = int(np.round(y))
            print(f"px:{px}, py:{py}, obstacle:{self.obstacles[py,px]}")

            if px < 0 or py < 0 or py >= self.obstacles.shape[0] or px >= self.obstacles.shape[1]:
                print(f"out of bounds, px:{px}, py:{py}, obstacles shape[1]:{self.obstacles.shape[1]}")
                return True 

            if self.obstacles[py,px] == 0:
                return True

        return False


    def is_goal_reached(self, current_node):
        dx = current_node.pos[0] - self.goal.pos[0]
        dy = current_node.pos[1] - self.goal.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        print(f"Distance to goal: {dist}")
        return dist < self.goal_dist

    def create_path(self, goal_index):
        path = [self.goal.pos]
        node = self.node_list[goal_index]
        
        while node != self.start:
            path.append(node.pos)
            node = node.parent
        
        path.append(node.pos)
        return path
