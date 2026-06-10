"""Sube los 100 testing frames + sus mascaras SAM3 (COCO) a Supervisely.

Lee el COCO ya generado (outputs/testing_100/annotations.json), crea un project
de Images con 3 ObjClass tipo Polygon (robot, orange_ball, green_floor), sube cada
imagen y convierte los poligonos COCO -> labels Supervisely. Corre en el pod.

Credenciales desde /workspace/.env: SUPERVISELY_API_TOKEN, SERVER_ADDRESS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import supervisely as sly

REPO = Path("/workspace/FutBotMX-UAQTeam")
COCO_PATH = REPO / "notebooks" / "fase_0" / "outputs" / "testing_100" / "annotations.json"
FRAMES_DIR = REPO / "data" / "testing_frames"
ENV_FILE = Path("/workspace/.env")

PROJECT_NAME = "FutBot_Testing_100"
DATASET_NAME = "testing_100"

COLORS = {
    "robot": [60, 130, 255],
    "orange_ball": [255, 100, 0],
    "green_floor": [50, 220, 70],
}


def load_env() -> None:
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()
    server = os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com")
    token = os.environ["SUPERVISELY_API_TOKEN"]
    print(f"server={server}")
    api = sly.Api(server, token)

    coco = json.loads(COCO_PATH.read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Team + workspace (usa el primero; crea workspace si no hay).
    team = api.team.get_list()[0]
    wss = api.workspace.get_list(team.id)
    ws = wss[0] if wss else api.workspace.create(team.id, "futbot", "")
    print(f"team={team.name!r}  workspace={ws.name!r}")

    # ObjClasses tipo Polygon (instance segmentation).
    obj_classes = {
        n: sly.ObjClass(n, sly.Polygon, color=COLORS.get(n, [255, 0, 0]))
        for n in cat_name.values()
    }
    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(list(obj_classes.values())))

    project = api.project.create(
        ws.id, PROJECT_NAME, type=sly.ProjectType.IMAGES, change_name_if_conflict=True
    )
    api.project.update_meta(project.id, meta.to_json())
    dataset = api.dataset.create(project.id, DATASET_NAME, change_name_if_conflict=True)
    print(f"project id={project.id} name={project.name!r}  dataset id={dataset.id}")

    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ok, skipped, failed = 0, 0, []
    for img in coco["images"]:
        path = FRAMES_DIR / img["file_name"]
        if not path.exists():
            skipped += 1
            continue
        try:
            info = api.image.upload_path(dataset.id, name=img["file_name"], path=str(path))
            h, w = img["height"], img["width"]
            labels = []
            for a in anns_by_img.get(img["id"], []):
                oc = obj_classes[cat_name[a["category_id"]]]
                for poly in a["segmentation"]:
                    ext = [
                        sly.PointLocation(row=int(poly[i + 1]), col=int(poly[i]))
                        for i in range(0, len(poly) - 1, 2)
                    ]
                    if len(ext) < 3:
                        continue
                    labels.append(sly.Label(sly.Polygon(ext), oc))
            ann = sly.Annotation(img_size=(h, w), labels=labels)
            api.annotation.upload_ann(info.id, ann)
            ok += 1
            if ok % 20 == 0:
                print(f"  {ok} subidas")
        except Exception as e:  # noqa: BLE001
            failed.append((img["file_name"], str(e)[:120]))

    print(f"\nListo: {ok} imgs subidas, {skipped} sin archivo, {len(failed)} fallaron.")
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")
    print(f"Project '{project.name}' (id={project.id}) en {server}")


if __name__ == "__main__":
    main()
