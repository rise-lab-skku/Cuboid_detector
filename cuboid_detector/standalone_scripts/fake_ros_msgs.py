from dataclasses import dataclass, field
from typing import List


@dataclass
class Header:
    # stamp: int
    frame_id: str = "L515_color_optical_frame"


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


@dataclass
class Pose:
    position: Point = Point()
    orientation: Quaternion = Quaternion()


@dataclass
class ObjectHypothesis:
    class_id: str = ""
    score: float = 0.0


@dataclass
class PoseWithCovariance:
    pose: Pose = Pose()
    covariance: List[float] = field(default_factory=list)


@dataclass
class ObjectHypothesisWithPose:
    # https://docs.ros.org/en/iron/p/vision_msgs/interfaces/msg/ObjectHypothesisWithPose.html
    hypothesis: ObjectHypothesis = ObjectHypothesis()
    pose: PoseWithCovariance = PoseWithCovariance()


@dataclass
class Detection3D:
    header: Header = Header()
    results: List[ObjectHypothesisWithPose] = field(default_factory=list)
    # bbox: BoundingBox3D
    # source_cloud: PointCloud2


@dataclass
class Detection3DArray:
    header: Header = Header()
    detections: List[Detection3D] = field(default_factory=list)
