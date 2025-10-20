# app/graphs.py
import math
import matplotlib
matplotlib.use("Agg")  # backend para servidor
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.path import Path
import numpy as np

# ===== pega aquí SIN CAMBIOS tu código de la gráfica =====
def _build_figure():
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Arc
    from matplotlib.path import Path
    import numpy as np

    def draw_causal_graph(
        pos, edges, signs=None, feedback_loops=None, fig_size=(28,22),
        padding=0.6, feedback_radius_px=1700, feedback_color='lightgray',
        vgap=3.0, arrow_ms=17, lw=1.4
    ):
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')

        macro_nodes = [
            "Emisiones","Dispersión atmosférica","Exposición poblacional",
            "Salud respiratoria","Respuesta social y regulatoria"
        ]
        font_macro_size = 16
        font_micro_size = 11

        text_objs = {}
        for name, (x,y) in pos.items():
            if name in macro_nodes:
                text_objs[name] = ax.text(
                    x, y, name, fontsize=font_macro_size,
                    ha='center', va='center', zorder=5, color="black",
                    fontname="Times New Roman", weight="bold"
                )
            else:
                text_objs[name] = ax.text(
                    x, y, name, fontsize=font_micro_size,
                    ha='center', va='center', zorder=5, color="black"
                )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        node_half = {}
        for name, txt in text_objs.items():
            bbox = txt.get_window_extent(renderer=renderer)
            inv = ax.transData.inverted()
            (x0, y0) = inv.transform((bbox.x0, bbox.y0))
            (x1, y1) = inv.transform((bbox.x1, bbox.y1))
            half_w = abs(x1 - x0) / 2.0 + padding
            half_h = abs(y1 - y0) / 2.0 + padding
            if name in macro_nodes:
                half_w *= 1.9; half_h *= 1.9
            node_half[name] = (half_w, half_h)

        def vertical_bezier_points(src, dst):
            (sx, sy) = pos[src]; (dx, dy) = pos[dst]
            (sw, sh) = node_half.get(src, (0.8,0.5))
            (dw, dh) = node_half.get(dst, (0.8,0.5))
            going_up = dy > sy
            sgn_src = 1 if going_up else -1
            p0 = (sx, sy + sgn_src*(sh + 0.25))
            c1 = (sx, sy + sgn_src*(sh + 0.25 + vgap))
            sgn_dst = -1 if going_up else 1
            p3 = (dx, dy + sgn_dst*(dh + 0.25))
            c2 = (dx, dy + sgn_dst*(dh + 0.25 + vgap))
            return p0, c1, c2, p3

        def add_arrow_bezier(p0, c1, c2, p3, color="black", z=3):
            verts = [p0, c1, c2, p3]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            arr = FancyArrowPatch(
                path=path, arrowstyle='-|>', mutation_scale=arrow_ms,
                linewidth=1.5, color=color, zorder=z, shrinkA=0, shrinkB=0, joinstyle="miter"
            )
            ax.add_patch(arr)
            return arr

        for (src, dst) in edges:
            if src not in pos or dst not in pos: continue
            p0, c1, c2, p3 = vertical_bezier_points(src, dst)
            add_arrow_bezier(p0, c1, c2, p3, color="black", z=3)
            sign = signs.get((src, dst), '+') if signs else '+'
            color = "darkred" if sign == "+" else ("blue" if sign == "-" else "gray")
            (dx, dy) = p3; (_, dh) = node_half.get(dst, (0.8, 0.5))
            going_up = dy < pos[dst][1]
            off_y = (dh + 0.9) * (1 if going_up else -1)
            ax.text(dx + 0.9, pos[dst][1] + off_y, sign,
                    fontsize=13, ha='center', va='center', zorder=6,
                    color=color, weight="bold")

        def add_feedback_circle_arrow(src, dst, offset_pixels=36):
            if src not in pos or dst not in pos: return
            x1, y1 = pos[src]; x2, y2 = pos[dst]
            src_disp = ax.transData.transform((x1, y1))
            dst_disp = ax.transData.transform((x2, y2))
            vx = dst_disp[0] - src_disp[0]; vy = dst_disp[1] - src_disp[1]
            dist_disp = math.hypot(vx, vy)
            if dist_disp < 2: return
            px = -vy/dist_disp; py = vx/dist_disp
            mid_disp = ((src_disp[0]+dst_disp[0])/2 + px*offset_pixels,
                        (src_disp[1]+dst_disp[1])/2 + py*offset_pixels)
            trans_inv = ax.transData.inverted()
            center_data = trans_inv.transform(mid_disp)
            radius_px = 1700
            width_data = abs(trans_inv.transform((mid_disp[0]+radius_px, mid_disp[1]))[0] - center_data[0])
            height_data = width_data
            arc = Arc(center_data, width=width_data, height=height_data,
                      theta1=0, theta2=320, linewidth=3.5,
                      color='lightgray', zorder=2.5)
            ax.add_patch(arc)
            end_angle = math.radians(320)
            ex = center_data[0] + (width_data/2)*math.cos(end_angle)
            ey = center_data[1] + (height_data/2)*math.sin(end_angle)
            arrow = FancyArrowPatch((ex-0.4, ey-0.3), (ex, ey),
                                    arrowstyle='-|>', color='lightgray',
                                    linewidth=2.8, zorder=3.5)
            ax.add_patch(arrow)

        pos_main = {
            "Emisiones": (0.0, 0.0), "Dispersión atmosférica": (0.0, 20.0),
            "Exposición poblacional": (25.0, 0.0), "Salud respiratoria": (0.0, -20.0),
            "Respuesta social y regulatoria": (-25.0, 0.0)
        }
        def generate_subnode_positions_circle(pos_main, children_dict, orbit_radius=8.5):
            pos = dict(pos_main)
            for parent, childs in children_dict.items():
                cx, cy = pos_main[parent]; n = len(childs)
                for i, child in enumerate(childs):
                    theta = 2*math.pi * i/n
                    pos[child] = (cx + orbit_radius*math.cos(theta),
                                  cy + orbit_radius*math.sin(theta))
            return pos

        children = {
            "Emisiones": ["Tipo_Fuente","Gas (E)","Tasa_Emisión","Ubicación (E)","Eficiencia_Control"],
            "Dispersión atmosférica": ["Gas (D)","Velocidad_Viento","Dirección_Viento","Temperatura","Humedad_Relativa","Concentración"],
            "Exposición poblacional": ["Zona (Exp)","Tamaño_Población","Tiempo_Exposición","Nivel_Exposición","Actividades_Aire_Libre"],
            "Salud respiratoria": ["Zona (Salud)","Casos_Asma","Casos_Bronquitis","Hospitalizaciones","Mortalidad","Atención_Médica_Disponible"],
            "Respuesta social y regulatoria": ["Tipo_Medida","Institución","Impacto_Estimado","Observaciones","Campañas_Sensibilización"]
        }
        pos_all = generate_subnode_positions_circle(pos_main, children)

        edges = [
            ("Emisiones","Dispersión atmosférica"),
            ("Dispersión atmosférica","Exposición poblacional"),
            ("Exposición poblacional","Salud respiratoria"),
            ("Salud respiratoria","Respuesta social y regulatoria"),
            ("Respuesta social y regulatoria","Emisiones"),
        ]
        for parent, childs in children.items():
            for c in childs: edges.append((c, parent))
        edges.extend([
            ("Tasa_Emisión","Concentración"), ("Velocidad_Viento","Concentración"),
            ("Dirección_Viento","Concentración"), ("Humedad_Relativa","Concentración"),
            ("Concentración","Nivel_Exposición"), ("Actividades_Aire_Libre","Nivel_Exposición"),
            ("Nivel_Exposición","Casos_Asma"), ("Nivel_Exposición","Casos_Bronquitis"),
            ("Atención_Médica_Disponible","Mortalidad"), ("Impacto_Estimado","Emisiones"),
            ("Tipo_Medida","Impacto_Estimado"), ("Institución","Impacto_Estimado"),
            ("Campañas_Sensibilización","Tipo_Medida"), ("Ubicación (E)","Zona (Exp)"),
            ("Zona (Exp)","Zona (Salud)")
        ])

        signs = {
            ("Emisiones","Dispersión atmosférica"): "+",
            ("Dispersión atmosférica","Exposición poblacional"): "+",
            ("Exposición poblacional","Salud respiratoria"): "+",
            ("Salud respiratoria","Respuesta social y regulatoria"): "+",
            ("Respuesta social y regulatoria","Emisiones"): "-",
        }

        feedback_loops = [
            ("Emisiones","Dispersión atmosférica"),
            ("Exposición poblacional","Salud respiratoria"),
            ("Salud respiratoria","Respuesta social y regulatoria"),
            ("Respuesta social y regulatoria","Emisiones")
        ]

        draw_causal_graph(
            pos_all, edges, signs=signs, feedback_loops=feedback_loops,
            feedback_color='lightgray', vgap=3.2, padding=0.7, arrow_ms=18, lw=1.5
        )
        return plt.gcf()

    return _build_figure()

def build_figure():
    """Punto de entrada público: devuelve matplotlib.figure.Figure"""
    return _build_figure()
