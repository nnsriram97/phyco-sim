# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubric color classes."""

import colorsys
from typing import NamedTuple, Tuple, Union
import random

class Color(NamedTuple):
  """Represents a color in terms of float values for RGBA between 0.0 and 1.0."""

  r: float
  g: float
  b: float
  a: float = 1.

  # Class attribute containing all predefined colors
  color_choices = {
        "aqua":    None,  # Will be populated after class definition
        "black":   None,
        "blue":    None,
        "fuchsia": None,
        "green":   None,
        "gray":    None,
        "lime":    None,
        "maroon":  None,
        "navy":    None,
        "olive":   None,
        "purple":  None,
        "red":     None,
        "silver":  None,
        "teal":    None,
        "white":   None,
        "yellow":  None,
        "aliceblue": None,
        "antiquewhite": None,
        "aquamarine": None,
        "azure": None,
        "beige": None,
        "bisque": None,
        "blanchedalmond": None,
        "blueviolet": None,
        "brown": None,
        "burlywood": None,
        "cadetblue": None,
        "chartreuse": None,
        "chocolate": None,
        "coral": None,
        "cornflowerblue": None,
        "cornsilk": None,
        "crimson": None,
        "cyan": None,
        "darkblue": None,
        "darkcyan": None,
        "darkgoldenrod": None,
        "darkgray": None,
        "darkgreen": None,
        "darkgrey": None,
        "darkkhaki": None,
        "darkmagenta": None,
        "darkolivegreen": None,
        "darkorange": None,
        "darkorchid": None,
        "darkred": None,
        "darksalmon": None,
        "darkseagreen": None,
        "darkslateblue": None,
        "darkslategray": None,
        "darkslategrey": None,
        "darkturquoise": None,
        "darkviolet": None,
        "deeppink": None,
        "deepskyblue": None,
        "dimgray": None,
        "dimgrey": None,
        "dodgerblue": None,
        "firebrick": None,
        "floralwhite": None,
        "forestgreen": None,
        "gainsboro": None,
        "ghostwhite": None,
        "gold": None,
        "goldenrod": None,
        "greenyellow": None,
        "grey": None,
        "honeydew": None,
        "hotpink": None,
        "indianred": None,
        "indigo": None,
        "ivory": None,
        "khaki": None,
        "lavender": None,
        "lavenderblush": None,
        "lawngreen": None,
        "lemonchiffon": None,
        "lightblue": None,
        "lightcoral": None,
        "lightcyan": None,
        "lightgoldenrodyellow": None,
        "lightgray": None,
        "lightgreen": None,
        "lightgrey": None,
        "lightpink": None,
        "lightsalmon": None,
        "lightseagreen": None,
        "lightskyblue": None,
        "lightslategray": None,
        "lightslategrey": None,
        "lightsteelblue": None,
        "lightyellow": None,
        "limegreen": None,
        "linen": None,
        "magenta": None,
        "mediumaquamarine": None,
        "mediumblue": None,
        "mediumorchid": None,
        "mediumpurple": None,
        "mediumseagreen": None,
        "mediumslateblue": None,
        "mediumspringgreen": None,
        "mediumturquoise": None,
        "mediumvioletred": None,
        "midnightblue": None,
        "mintcream": None,
        "mistyrose": None,
        "moccasin": None,
        "navajowhite": None,
        "oldlace": None,
        "olivedrab": None,
        "orange": None,
        "orangered": None,
        "orchid": None,
        "palegoldenrod": None,
        "palegreen": None,
        "paleturquoise": None,
        "palevioletred": None,
        "papayawhip": None,
        "peachpuff": None,
        "peru": None,
        "pink": None,
        "plum": None,
        "powderblue": None,
        "rebeccapurple": None,
        "rosybrown": None,
        "royalblue": None,
        "saddlebrown": None,
        "salmon": None,
        "sandybrown": None,
        "seagreen": None,
        "seashell": None,
        "sienna": None,
        "skyblue": None,
        "slateblue": None,
        "slategray": None,
        "slategrey": None,
        "snow": None,
        "springgreen": None,
        "steelblue": None,
        "tan": None,
        "thistle": None,
        "tomato": None,
        "turquoise": None,
        "violet": None,
        "wheat": None,
        "whitesmoke": None,
        "yellowgreen": None,
        # Additional darker colors for better coverage
        "charcoal": None,
        "gunmetal": None,
        "darkcharcoal": None,
        "almostblack": None,
        "deepnavyblue": None,
        "darkforestgreen": None,
        "darkbrown": None,
        "deepmaroon": None,
        "darkpurple": None,
    }

  @property
  def rgb(self):
    return self.r, self.g, self.b

  @property
  def hsv(self):
    return colorsys.rgb_to_hsv(self.r, self.g, self.b)

  @property
  def hexstr(self):
    r, g, b, a = [int(x * 255) for x in iter(self)]
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

  @property
  def hexstr_short(self):
    r, g, b, a = [int(x * 15) for x in iter(self)]
    return f"#{r:01x}{g:01x}{b:01x}{a:01x}"

  @classmethod
  def from_hsv(cls, h: float, s: float, v: float, alpha=1.0):
    if not 0 <= h <= 1:
      raise ValueError(f"Hue has to be between 0.0 and 1.0 (was {h})")
    if not 0 <= s <= 1:
      raise ValueError(f"Saturation has to be between 0.0 and 1.0 (was {s})")
    if not 0 <= v <= 1:
      raise ValueError(f"Value has to be between 0.0 and 1.0 (was {v})")
    return cls(*colorsys.hsv_to_rgb(h, s, v), a=alpha)

  @classmethod
  def from_hexint(cls, hexint: int, alpha: float = 1.0):
    """Create a Color instance from a hex integer like 0xaaff33 and an optional alpha value."""
    if not 0 <= hexint <= 0xffffff:
      raise ValueError(f"hexint not [0x000000 ... 0xffffff] (was 0x{hexint:06x})")
    if not 0. <= alpha <= 1.0:
      raise ValueError(f"alpha has to be between 0.0 and 1.0 (was {alpha})")
    b = hexint & 255
    g = (hexint >> 8) & 255
    r = (hexint >> 16) & 255
    return cls(r / 255.0, g / 255.0, b / 255.0, alpha)

  @classmethod
  def from_hexstr(cls, hexstr: str):
    """Create a Color instance from a hex string like #ffaa22 or #11aa88ff.

    Supports both long and short form (i.e. #ffffff is the same as #fff), and also an optional
    alpha value (e.g. #112233ff or #123f).
    """
    if hexstr[0] == "#":  # get rid of leading #
      hexstr = hexstr[1:]
    if len(hexstr) == 3:
      r = int(hexstr[0], 16) / 15.
      g = int(hexstr[1], 16) / 15.
      b = int(hexstr[2], 16) / 15.
      return cls(r, g, b)
    elif len(hexstr) == 4:
      r = int(hexstr[0], 16) / 15.
      g = int(hexstr[1], 16) / 15.
      b = int(hexstr[2], 16) / 15.
      a = int(hexstr[3], 16) / 15.
      return cls(r, g, b, a)
    elif len(hexstr) == 6:
      r = int(hexstr[0:2], 16) / 255.0
      g = int(hexstr[2:4], 16) / 255.0
      b = int(hexstr[4:6], 16) / 255.0
      return cls(r, g, b)
    elif len(hexstr) == 8:
      r = int(hexstr[0:2], 16) / 255.0
      g = int(hexstr[2:4], 16) / 255.0
      b = int(hexstr[4:6], 16) / 255.0
      a = int(hexstr[6:8], 16) / 255.0
      return cls(r, g, b, a)
    else:
      raise ValueError("invalid color hex string")

  @classmethod
  def from_name(cls, name: str):
    return cls.color_choices[name.lower()]

  @classmethod
  def random_color(cls):
    return cls.from_name(random.choice(list(cls.color_choices.keys())))


def get_color(color: Union[int, str, Tuple]) -> Color:
  if isinstance(color, str):
    if color.startswith("#"):
      return Color.from_hexstr(color)
    else:
      return Color.from_name(color)
  elif isinstance(color, int):
    return Color.from_hexint(color)
  else:
    return Color(*color)


# Populate the color_choices dictionary after class definition
Color.color_choices.update({
    "aqua":    Color.from_hexstr("#00ffff"),
    "black":   Color.from_hexstr("#000000"),
    "blue":    Color.from_hexstr("#0000ff"),
    "fuchsia": Color.from_hexstr("#ff00ff"),
    "green":   Color.from_hexstr("#008000"),
    "gray":    Color.from_hexstr("#808080"),
    "lime":    Color.from_hexstr("#00ff00"),
    "maroon":  Color.from_hexstr("#800000"),
    "navy":    Color.from_hexstr("#000080"),
    "olive":   Color.from_hexstr("#808000"),
    "purple":  Color.from_hexstr("#800080"),
    "red":     Color.from_hexstr("#ff0000"),
    "silver":  Color.from_hexstr("#c0c0c0"),
    "teal":    Color.from_hexstr("#008080"),
    "white":   Color.from_hexstr("#ffffff"),
    "yellow":  Color.from_hexstr("#ffff00"),
    "aliceblue": Color.from_hexstr("#f0f8ff"),
    "antiquewhite": Color.from_hexstr("#faebd7"),
    "aquamarine": Color.from_hexstr("#7fffd4"),
    "azure": Color.from_hexstr("#f0ffff"),
    "beige": Color.from_hexstr("#f5f5dc"),
    "bisque": Color.from_hexstr("#ffe4c4"),
    "blanchedalmond": Color.from_hexstr("#ffebcd"),
    "blueviolet": Color.from_hexstr("#8a2be2"),
    "brown": Color.from_hexstr("#a52a2a"),
    "burlywood": Color.from_hexstr("#deb887"),
    "cadetblue": Color.from_hexstr("#5f9ea0"),
    "chartreuse": Color.from_hexstr("#7fff00"),
    "chocolate": Color.from_hexstr("#d2691e"),
    "coral": Color.from_hexstr("#ff7f50"),
    "cornflowerblue": Color.from_hexstr("#6495ed"),
    "cornsilk": Color.from_hexstr("#fff8dc"),
    "crimson": Color.from_hexstr("#dc143c"),
    "cyan": Color.from_hexstr("#00ffff"),
    "darkblue": Color.from_hexstr("#00008b"),
    "darkcyan": Color.from_hexstr("#008b8b"),
    "darkgoldenrod": Color.from_hexstr("#b8860b"),
    "darkgray": Color.from_hexstr("#a9a9a9"),
    "darkgreen": Color.from_hexstr("#006400"),
    "darkgrey": Color.from_hexstr("#a9a9a9"),
    "darkkhaki": Color.from_hexstr("#bdb76b"),
    "darkmagenta": Color.from_hexstr("#8b008b"),
    "darkolivegreen": Color.from_hexstr("#556b2f"),
    "darkorange": Color.from_hexstr("#ff8c00"),
    "darkorchid": Color.from_hexstr("#9932cc"),
    "darkred": Color.from_hexstr("#8b0000"),
    "darksalmon": Color.from_hexstr("#e9967a"),
    "darkseagreen": Color.from_hexstr("#8fbc8f"),
    "darkslateblue": Color.from_hexstr("#483d8b"),
    "darkslategray": Color.from_hexstr("#2f4f4f"),
    "darkslategrey": Color.from_hexstr("#2f4f4f"),
    "darkturquoise": Color.from_hexstr("#00ced1"),
    "darkviolet": Color.from_hexstr("#9400d3"),
    "deeppink": Color.from_hexstr("#ff1493"),
    "deepskyblue": Color.from_hexstr("#00bfff"),
    "dimgray": Color.from_hexstr("#696969"),
    "dimgrey": Color.from_hexstr("#696969"),
    "dodgerblue": Color.from_hexstr("#1e90ff"),
    "firebrick": Color.from_hexstr("#b22222"),
    "floralwhite": Color.from_hexstr("#fffaf0"),
    "forestgreen": Color.from_hexstr("#228b22"),
    "gainsboro": Color.from_hexstr("#dcdcdc"),
    "ghostwhite": Color.from_hexstr("#f8f8ff"),
    "gold": Color.from_hexstr("#ffd700"),
    "goldenrod": Color.from_hexstr("#daa520"),
    "greenyellow": Color.from_hexstr("#adff2f"),
    "grey":    Color.from_hexstr("#808080"),
    "honeydew": Color.from_hexstr("#f0fff0"),
    "hotpink": Color.from_hexstr("#ff69b4"),
    "indianred": Color.from_hexstr("#cd5c5c"),
    "indigo": Color.from_hexstr("#4b0082"),
    "ivory": Color.from_hexstr("#fffff0"),
    "khaki": Color.from_hexstr("#f0e68c"),
    "lavender": Color.from_hexstr("#e6e6fa"),
    "lavenderblush": Color.from_hexstr("#fff0f5"),
    "lawngreen": Color.from_hexstr("#7cfc00"),
    "lemonchiffon": Color.from_hexstr("#fffacd"),
    "lightblue": Color.from_hexstr("#add8e6"),
    "lightcoral": Color.from_hexstr("#f08080"),
    "lightcyan": Color.from_hexstr("#e0ffff"),
    "lightgoldenrodyellow": Color.from_hexstr("#fafad2"),
    "lightgray": Color.from_hexstr("#d3d3d3"),
    "lightgreen": Color.from_hexstr("#90ee90"),
    "lightgrey": Color.from_hexstr("#d3d3d3"),
    "lightpink": Color.from_hexstr("#ffb6c1"),
    "lightsalmon": Color.from_hexstr("#ffa07a"),
    "lightseagreen": Color.from_hexstr("#20b2aa"),
    "lightskyblue": Color.from_hexstr("#87cefa"),
    "lightslategray": Color.from_hexstr("#778899"),
    "lightslategrey": Color.from_hexstr("#778899"),
    "lightsteelblue": Color.from_hexstr("#b0c4de"),
    "lightyellow": Color.from_hexstr("#ffffe0"),
    "limegreen": Color.from_hexstr("#32cd32"),
    "linen": Color.from_hexstr("#faf0e6"),
    "magenta": Color.from_hexstr("#ff00ff"),
    "mediumaquamarine": Color.from_hexstr("#66cdaa"),
    "mediumblue": Color.from_hexstr("#0000cd"),
    "mediumorchid": Color.from_hexstr("#ba55d3"),
    "mediumpurple": Color.from_hexstr("#9370db"),
    "mediumseagreen": Color.from_hexstr("#3cb371"),
    "mediumslateblue": Color.from_hexstr("#7b68ee"),
    "mediumspringgreen": Color.from_hexstr("#00fa9a"),
    "mediumturquoise": Color.from_hexstr("#48d1cc"),
    "mediumvioletred": Color.from_hexstr("#c71585"),
    "midnightblue": Color.from_hexstr("#191970"),
    "mintcream": Color.from_hexstr("#f5fffa"),
    "mistyrose": Color.from_hexstr("#ffe4e1"),
    "moccasin": Color.from_hexstr("#ffe4b5"),
    "navajowhite": Color.from_hexstr("#ffdead"),
    "oldlace": Color.from_hexstr("#fdf5e6"),
    "olivedrab": Color.from_hexstr("#6b8e23"),
    "orange": Color.from_hexstr("#ffa500"),
    "orangered": Color.from_hexstr("#ff4500"),
    "orchid": Color.from_hexstr("#da70d6"),
    "palegoldenrod": Color.from_hexstr("#eee8aa"),
    "palegreen": Color.from_hexstr("#98fb98"),
    "paleturquoise": Color.from_hexstr("#afeeee"),
    "palevioletred": Color.from_hexstr("#db7093"),
    "papayawhip": Color.from_hexstr("#ffefd5"),
    "peachpuff": Color.from_hexstr("#ffdab9"),
    "peru": Color.from_hexstr("#cd853f"),
    "pink": Color.from_hexstr("#ffc0cb"),
    "plum": Color.from_hexstr("#dda0dd"),
    "powderblue": Color.from_hexstr("#b0e0e6"),
    "rebeccapurple": Color.from_hexstr("#663399"),
    "rosybrown": Color.from_hexstr("#bc8f8f"),
    "royalblue": Color.from_hexstr("#4169e1"),
    "saddlebrown": Color.from_hexstr("#8b4513"),
    "salmon": Color.from_hexstr("#fa8072"),
    "sandybrown": Color.from_hexstr("#f4a460"),
    "seagreen": Color.from_hexstr("#2e8b57"),
    "seashell": Color.from_hexstr("#fff5ee"),
    "sienna": Color.from_hexstr("#a0522d"),
    "skyblue": Color.from_hexstr("#87ceeb"),
    "slateblue": Color.from_hexstr("#6a5acd"),
    "slategray": Color.from_hexstr("#708090"),
    "slategrey": Color.from_hexstr("#708090"),
    "snow": Color.from_hexstr("#fffafa"),
    "springgreen": Color.from_hexstr("#00ff7f"),
    "steelblue": Color.from_hexstr("#4682b4"),
    "tan": Color.from_hexstr("#d2b48c"),
    "thistle": Color.from_hexstr("#d8bfd8"),
    "tomato": Color.from_hexstr("#ff6347"),
    "turquoise": Color.from_hexstr("#40e0d0"),
    "violet": Color.from_hexstr("#ee82ee"),
    "wheat": Color.from_hexstr("#f5deb3"),
    "whitesmoke": Color.from_hexstr("#f5f5f5"),
    "yellowgreen": Color.from_hexstr("#9acd32"),
    # Additional darker colors for better coverage
    "charcoal": Color.from_hexstr("#36454f"),
    "gunmetal": Color.from_hexstr("#2a3439"),
    "darkcharcoal": Color.from_hexstr("#333333"),
    "almostblack": Color.from_hexstr("#1a1a1a"),
    "deepnavyblue": Color.from_hexstr("#1e2952"),
    "darkforestgreen": Color.from_hexstr("#013220"),
    "darkbrown": Color.from_hexstr("#654321"),
    "deepmaroon": Color.from_hexstr("#722f37"),
    "darkpurple": Color.from_hexstr("#301934"),
})
