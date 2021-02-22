from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

# Note that the class name MUST be a camel-case version of the file name
class DislPickUpBlueCup(Task):

    def init_task(self) -> None:
        self.cup = Shape('cup')
        self.cup_visual = Shape('cup_visual')
        self.boundary = SpawnBoundary([Shape('boundary')])
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.cup])
        self.register_success_conditions([
            DetectedCondition(self.cup, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.cup),
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = 0
        target_color_name, target_rgb = colors[4]  # gets ('blue', (0.0,0.0,1.0))


        self.cup_visual.set_color(target_rgb)

        self.boundary.clear()
        self.boundary.sample(self.success_sensor, min_distance=0.1)

        return ['pick up the %s cup' % target_color_name,
                'grasp the %s cup and lift it' % target_color_name,
                'lift the %s cup' % target_color_name]

    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -0.01], [0.0, 0.0, -0.01]


    def boundary_root(self) -> Object:
        """An object that should act as the task root when randomly placing.

        Sometimes the task can involve a large appliance (e.g. dish washer)
        which cannot be placed within the task workspace. This allows us to
        define a proxy root (e.g. a plane in front of the appliance) that
        would allow the task to be placed inside the workspace.

        :return: The PyRep Object that will act as the root.
        """
        return self.get_base()



    def is_static_workspace(self) -> bool:
        return True

    def reward(self): # -> Union[float, None]:
        """Allows the user to customise the task and add reward shaping."""
        return None

    def variation_count(self) -> int:
        return 1
