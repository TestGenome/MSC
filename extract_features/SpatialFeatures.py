from pysc2.lib.features import *
from pysc2.lib import stopwatch

sw = stopwatch.sw

class ScreenFeatures(collections.namedtuple("ScreenFeatures", ["height_map", "visibility_map",
                    "creep", "power", "player_relative", "unit_type", "unit_density", "unit_density_aa"])):
  """The set of screen feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette, clip) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=ScreenFeatures._fields.index(name),
          name=name,
          layer_set="renders",
          full_name="screen " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=clip)
    return super(ScreenFeatures, cls).__new__(cls, **feats)


class MinimapFeatures(collections.namedtuple("MinimapFeatures", [
    "height_map", "visibility_map", "creep", "player_relative"])):
  """The set of minimap feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=MinimapFeatures._fields.index(name),
          name=name,
          layer_set="minimap_renders",
          full_name="minimap " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=False)
    return super(MinimapFeatures, cls).__new__(cls, **feats)

SCREEN_FEATURES = ScreenFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter, False),
    visibility_map=(4, FeatureType.CATEGORICAL,
                    colors.VISIBILITY_PALETTE, False),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE, False),
    power=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE, False),
    unit_type=(1962, FeatureType.CATEGORICAL, colors.unit_type, False),
    unit_density=(16, FeatureType.SCALAR, colors.hot, False),
    unit_density_aa=(256, FeatureType.SCALAR, colors.hot, False),
)

MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter),
    visibility_map=(4, FeatureType.CATEGORICAL, colors.VISIBILITY_PALETTE),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE)
)

class SpatialFeatures(Features):

    def __init__(self, game_info, agent_interface_format=None, map_name=None, **kwargs):
        """Construct a Features object using data extracted from game info.

        Args:
            game_info: A `sc_pb.ResponseGameInfo` from the game.
            agent_interface_format: an optional AgentInterfaceFormat.
            map_name: an optional map name, which overrides the one in game_info.
            **kwargs: Anything else is passed through to AgentInterfaceFormat. It's an
                error to send any kwargs if you pass an agent_interface_format.

        Returns:
            A features object matching the specified parameterisation.

        Raises:
            ValueError: if you pass both agent_interface_format and kwargs.
            ValueError: if you pass an agent_interface_format that doesn't match
                game_info's resolutions.
        """
        if not map_name:
            map_name = game_info.map_name

        if game_info.options.HasField("feature_layer"):
            fl_opts = game_info.options.feature_layer
            feature_dimensions = Dimensions(
                screen=(fl_opts.resolution.x, fl_opts.resolution.y),
                minimap=(fl_opts.minimap_resolution.x, fl_opts.minimap_resolution.y))
            camera_width_world_units = game_info.options.feature_layer.width
        else:
            feature_dimensions = None
            camera_width_world_units = None

        if game_info.options.HasField("render"):
            rgb_opts = game_info.options.render
            rgb_dimensions = Dimensions(
                screen=(rgb_opts.resolution.x, rgb_opts.resolution.y),
                minimap=(rgb_opts.minimap_resolution.x, rgb_opts.minimap_resolution.y))
        else:
            rgb_dimensions = None

        map_size = game_info.start_raw.map_size

        requested_races = {
            info.player_id: info.race_requested for info in game_info.player_info
            if info.type != sc_pb.Observer}

        if agent_interface_format:
            if kwargs:
                raise ValueError(
                    "Either give an agent_interface_format or kwargs, not both.")
            aif = agent_interface_format
            if (aif.rgb_dimensions != rgb_dimensions or
                aif.feature_dimensions != feature_dimensions or
                (feature_dimensions and
                aif.camera_width_world_units != camera_width_world_units)):
                raise ValueError("""
            The supplied agent_interface_format doesn't match the resolutions computed from
            the game_info:
            rgb_dimensions: %s != %s
            feature_dimensions: %s != %s
            camera_width_world_units: %s != %s
            """ % (aif.rgb_dimensions, rgb_dimensions,
                aif.feature_dimensions, feature_dimensions,
                aif.camera_width_world_units, camera_width_world_units))
        else:
            agent_interface_format = AgentInterfaceFormat(
                feature_dimensions=feature_dimensions,
                rgb_dimensions=rgb_dimensions,
                camera_width_world_units=camera_width_world_units,
                **kwargs)

        super().__init__(agent_interface_format=agent_interface_format,
            map_size=map_size,
            map_name=map_name,
            requested_races=requested_races)

    def observation_spec(self):
        """The observation spec for the SC2 environment.
        Returns:
          The dict of observation names to their tensor shapes. Shapes with a 0 can
          vary in length, for example the number of valid actions depends on which
          units you have selected.
        """
        return {
            "screen": (len(SCREEN_FEATURES),
                       self._screen_size_px.y,
                       self._screen_size_px.x),
            "minimap": (len(MINIMAP_FEATURES),
                        self._minimap_size_px.y,
                        self._minimap_size_px.x),
            "player": (11,),
            "score": (13,)
        }

    @sw.decorate
    def transform_obs(self, obs):
        """Render some SC2 observations into something an agent can handle."""
        out = {}

        with sw("feature_layers"):
            out["screen"] = np.stack(
                f.unpack(obs)/f.scale for f in SCREEN_FEATURES).astype(np.float32, copy=False)
            out["minimap"] = np.stack(
                f.unpack(obs)/f.scale for f in MINIMAP_FEATURES).astype(np.float32, copy=False)

        out["player"] = np.array([
            obs.game_loop - 1,
            obs.player_common.minerals,
            obs.player_common.vespene,
            obs.player_common.food_used,
            obs.player_common.food_cap,
            obs.player_common.food_army,
            obs.player_common.food_workers,
            obs.player_common.idle_worker_count,
            obs.player_common.army_count,
            obs.player_common.warp_gate_count,
            obs.player_common.larva_count,
        ], dtype=np.int32)

        out["score"] = np.array([
            obs.score.score,
            obs.score.score_details.idle_production_time,
            obs.score.score_details.idle_worker_time,
            obs.score.score_details.total_value_units,
            obs.score.score_details.total_value_structures,
            obs.score.score_details.killed_value_units,
            obs.score.score_details.killed_value_structures,
            obs.score.score_details.collected_minerals,
            obs.score.score_details.collected_vespene,
            obs.score.score_details.collection_rate_minerals,
            obs.score.score_details.collection_rate_vespene,
            obs.score.score_details.spent_minerals,
            obs.score.score_details.spent_vespene,
        ], dtype=np.int32)

        return out