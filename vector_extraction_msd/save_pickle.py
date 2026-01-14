import pickle

def classify_and_save_pickle(polygons, output_pickle):
    """
    Ask user how many polygons are stairs and save Swiss-format pickle.
    
    Args:
        polygons (list): list of polygons (each is list of (x,y) tuples)
        output_pickle (str): path to save pickle file
    """

    print("\n=== STAIR CLASSIFICATION ===")
    num_polys = len(polygons)

    # Ask user how many are stairs
    while True:
        try:
            n_stairs = int(input(f"How many stair polygons? (0–{num_polys}): ").strip())
            if 0 <= n_stairs <= num_polys:
                break
            print("Invalid number. Try again.")
        except:
            print("Enter a valid integer.")

    # Create Swiss-format structure
    floorplan_data = {"floor_plan": []}

    for idx, poly in enumerate(polygons):
        room_type = 1 if idx < n_stairs else 0  # stairs first, others apartments
        floorplan_data["floor_plan"].append({
            "polygon": poly,
            "room_type": room_type
        })

    # Save to pickle
    with open(output_pickle, "wb") as f:
        pickle.dump(floorplan_data, f)

    print(f"\n✅ Pickle saved: {output_pickle}")
    print(f"   Stair polygons: 0 → {n_stairs-1 if n_stairs>0 else 'None'}")
    print(f"   Apartment polygons: {n_stairs} → {num_polys-1}")
