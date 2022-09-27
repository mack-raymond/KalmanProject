from dash import (
    html,
)
import dash
import dash_bootstrap_components as dbc

pages = dash.page_registry
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Home",
                href=pages["pages.home"]["relative_path"],
            )
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(
                    "Tools",
                    href=pages["pages.tools"]["relative_path"],
                ),
                dbc.DropdownMenuItem(
                    "Contact",
                    href=pages["pages.contact"]["relative_path"],
                ),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Progress",
    brand_href="#",
    color="primary",
)
content = dash.page_container

children = [
    navbar,
    content,
]
layout = html.Div(children)
