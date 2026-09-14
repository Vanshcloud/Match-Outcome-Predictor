"""The vocabulary: what a match, a forecast and a competition *are* here.

Value types and nothing else. No HTTP, no file, no Streamlit call, no
provider — a module here can be imported by a test, by a view, by a provider,
and one day by a different front end, and it will behave the same in all four.

That is the property the layer exists for. Everything above it is replaceable:
:mod:`dashboard.views` is Streamlit today and could be a React client reading
the API tomorrow, and :mod:`dashboard.providers` is a historical reader and
several live feeds. Neither can change what a
:class:`~dashboard.domain.match.Fixture` means, because neither is imported
from here.
"""

from dashboard.domain.match import Fixture, MatchStatus, Prediction

__all__ = ["Fixture", "MatchStatus", "Prediction"]
