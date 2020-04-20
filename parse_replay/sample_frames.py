from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from absl import app
from absl import flags
from tqdm import tqdm

from google.protobuf.json_format import Parse

from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='../high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replays', default='../parsed_replays',
                    help='Path for parsed actions')
flags.DEFINE_string(name='infos_path', default='../replays_infos',
                    help='Paths for infos of replays')
flags.DEFINE_integer(name='step_mul', default=8,
                     help='step size')
flags.DEFINE_integer(name='skip', default=96,
                     help='# of skipped frames')

def sample_frames(action_path):
    agent_intf = features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=(1,1), minimap=(1,1)))
    feat = features.Features(agent_intf)

    with open(action_path) as f:
        actions = json.load(f)

    frame_id = 0
    result_frames = []
    for action_step in actions: # Get actions performed since previous observed frame
        frame_id += FLAGS.step_mul # Advance to current frame
        action_name = None
        for action_str in action_step: # Search all actions from step
            action = Parse(action_str, sc_pb.Action())
            try:
                func_id = feat.reverse_action(action).function
                func_name = FUNCTIONS[func_id].name
                if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'Cancel', 'Halt', 'Stop'}: # Macro action found in step
                    action_name = func_name
                    break # Macro step found, no need to process further actions from this step
            except:
                pass 
        if (action_name is not None) or ((frame_id % FLAGS.skip) == 0): # This is a macro step or fixed recording step
            result_frames.append(frame_id)

    return result_frames

def sample_frames(replay_path, action_path, sampled_frame_path):
    replay_info = os.path.join(FLAGS.infos_path, replay_path)
    if not os.path.isfile(replay_info):
        return
    with open(replay_info) as f:
        info = json.load(f)

    result = []
    proto = Parse(info['info'], sc_pb.ResponseReplayInfo())
    for p in proto.player_info: # Sample actions taken by each player
        player_id = p.player_info.player_id
        race = common_pb.Race.Name(p.player_info.race_actual)

        action_file = os.path.join(action_path, race, '{}@{}'.format(player_id, replay_path)) 
        if not os.path.isfile(action_file): # Skip replays where actions haven't been extracted yet
            print('Unable to locate', action_file)
            return

        result.append(sample_frames_from_player(action_file)) # Get the frames where each player took a macro action

    assert len(result) == 2
    sampled_actions = sorted(set(result[0]) | set(result[1])) # Collect all frames where either player took a macro action

    with open(os.path.join(sampled_frame_path, replay_path), 'w') as f:
        json.dump(sampled_actions, f)

def main(argv):
    with open(FLAGS.hq_replay_set) as f:
        replay_list = json.load(f)
    replay_list = sorted([p for p, _ in replay_list])

    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    sampled_frame_path = os.path.join(FLAGS.parsed_replays, 'SampledFrames', race_vs_race)
    if not os.path.isdir(sampled_frame_path):
        os.makedirs(sampled_frame_path)
    action_path = os.path.join(FLAGS.parsed_replays, 'Actions', race_vs_race)

    pbar = tqdm(total=len(replay_list), desc='#Replay')
    for replay_path in replay_list: # Extract macro frames from every replay
        sample_frames(os.path.basename(replay_path), action_path, sampled_frame_path)
        pbar.update()

if __name__ == '__main__':
    app.run(main)
