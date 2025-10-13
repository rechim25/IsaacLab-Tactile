# Peg Insert Manager-Based Environment

This document describes the new manager-based implementation of the Peg Insert task, which enables:
1. **Teleoperation with SpaceMouse** - Record human demonstrations
2. **SkillGen compatibility** - Generate additional demonstrations automatically

## Environment Details

### Registered Environments

The following environments have been created:

1. **Isaac-Peg-Insert-Franka-v0** - Joint position control
2. **Isaac-Peg-Insert-Franka-IK-Rel-v0** - Inverse kinematics with relative pose control (for teleoperation)
3. **Isaac-Peg-Insert-Franka-Play-v0** - Play version with 50 environments
4. **Isaac-Peg-Insert-Franka-IK-Rel-Play-v0** - IK Play version with 50 environments

### File Structure

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/peg_insert/
├── __init__.py                           # Environment registration
├── peg_insert_env_cfg.py                 # Base environment configuration
├── mdp/                                   # MDP-specific implementations
│   ├── __init__.py
│   ├── observations.py                    # Custom observations
│   └── terminations.py                    # Success/failure conditions
├── config/                                # Robot-specific configurations
│   └── franka/
│       ├── peg_insert_joint_pos_env_cfg.py    # Joint position control
│       └── peg_insert_ik_rel_env_cfg.py       # IK control for teleoperation
└── agents/                                # Agent configurations (for future use)
```

## Usage

### 1. Teleoperation with Keyboard

Record demonstrations using keyboard controls:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-v0 \
    --num_envs 1 \
    --teleop_device keyboard
```

**Keyboard Controls:**
- `W/S` - Move forward/backward
- `A/D` - Move left/right  
- `Q/E` - Move up/down
- `Z/X` - Rotate around X-axis
- `T/G` - Rotate around Y-axis
- `C/V` - Rotate around Z-axis
- `K` - Open gripper
- `L` - Close gripper

### 2. Teleoperation with SpaceMouse

For smoother demonstrations (recommended for SkillGen):

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-v0 \
    --num_envs 1 \
    --teleop_device spacemouse
```

**Note:** If SpaceMouse is not detected, grant permissions:
```bash
sudo chmod 666 /dev/hidraw<#>  # where <#> is your device index
```

To find the device index:
```bash
ls -l /dev/hidraw*
cat /sys/class/hidraw/hidraw<#>/device/uevent
```

### 3. Recording Demonstrations

To record demonstrations for imitation learning:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-v0 \
    --num_demos 50 \
    --teleop_device spacemouse
```

## Key Features

### Scene Configuration
- **Robot**: Franka Panda with high-PD gains for better IK tracking
- **Peg**: 8mm diameter peg (held by gripper)
- **Hole**: 8mm hole (fixed on table)
- **Table**: Seattle Lab table

### Observations
The policy observes:
- Joint positions and velocities (relative)
- End-effector position and orientation
- Peg position and orientation (in robot frame)
- Hole position and orientation (in robot frame)  
- Relative position from peg to hole
- Previous actions

### Actions
- **Arm**: 6-DOF differential IK control (position + orientation)
- **Gripper**: Binary open/close command

### Success Criteria
The task succeeds when:
1. Peg is centered over hole (XY distance < 2.5mm)
2. Peg has descended to target depth (Z < 4% of hole height)

### Termination Conditions
The episode terminates when:
- Task succeeds (peg inserted)
- Peg drops below table (failure)
- Time limit reached (10 seconds)

## Next Steps: SkillGen Integration

To enable SkillGen for automated demonstration generation, you need to:

1. **Create a SkillGen-specific configuration** similar to `stack_ik_rel_env_cfg_skillgen.py`
2. **Define subtasks** for the peg insertion task (e.g., "grasp", "approach", "insert")
3. **Add subtask observations** to detect when each subtask is active
4. **Create a Mimic environment** that extends `ManagerBasedRLMimicEnv`

Reference the stack cube task implementation at:
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_env_cfg_skillgen.py`
- `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py`

## Comparison with Direct Workflow

| Feature | Direct Workflow | Manager-Based |
|---------|----------------|---------------|
| Task Design | Code-based | Configuration-based |
| Observations | Custom methods | Observation terms |
| Actions | Custom logic | Action terms |
| Teleoperation | ❌ Not supported | ✅ Supported |
| SkillGen | ❌ Not supported | ✅ Supported |
| Modularity | Lower | Higher |

## Testing

To verify the environment works:

```bash
# Test with random agent  
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Peg-Insert-Franka-IK-Rel-v0 --num_envs 16

# Test with zero agent
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Peg-Insert-Franka-IK-Rel-v0 --num_envs 16
```

## Troubleshooting

### AttributeError: module has no attribute 'ee_frame_pos'

This error occurs if the custom observation functions are not properly imported. The fix is to add these functions to the mdp/observations.py file:
- `ee_frame_pos()` - Returns end-effector position from FrameTransformer
- `ee_frame_quat()` - Returns end-effector orientation from FrameTransformer  
- `gripper_pos()` - Returns gripper joint positions

These functions are already included in the implementation.

### "Found an articulation root" error

If you get an error like:
```
Found an articulation root when resolving '/World/envs/env_.*/Peg' for rigid objects
```

This means the USD files contain articulation roots but we're spawning them as rigid objects. The fix is to disable articulation in the spawn config:

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    articulation_enabled=False,
)
```

This is already included in the implementation for both peg and hole objects.

### AttributeError: 'NoneType' object has no attribute '__dict__'

If you see this error, it means some manager configurations are set to `None`. The fix is to define empty config classes:

```python
@configclass
class EventCfg:
    """Configuration for events (empty for now)."""
    pass
```

This is already included in the implementation.

## References

- [Official Teleoperation Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)
- [SkillGen Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)
- Stack Cube example: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/`

