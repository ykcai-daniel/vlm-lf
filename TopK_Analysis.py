import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def top_k_results(data, k:int, run_type:str):
    if run_type == 'image':
        all_results = []

        for frame in data['result']:
            if 'scores' in frame:
                for score, box in zip(frame['scores'], frame['boxes']):
                    all_results.append((frame['frame'], score, box))

        all_results.sort(key=lambda x: x[1], reverse=True)
        top_k = all_results[:k]

        new_data = [{'frame': i} for i in range(max([frame for frame, _, _ in all_results]) + 1)]

        for frame, score, box in top_k:
            if 'scores' not in new_data[frame]:
                new_data[frame]['scores'] = []
                new_data[frame]['boxes'] = []

            new_data[frame]['scores'].append(score)
            new_data[frame]['boxes'].append(box)
    elif run_type == 'lang':
        all_results = {}

        for frame in data['result']:
            if 'scores' in frame:
                for score, box, label in zip(frame['scores'], frame['boxes'], frame['labels']):
                    if label not in all_results:
                        all_results[label] = []
                    all_results[label].append((frame['frame'], score, box))

        for label, results in all_results.items():
            results.sort(key=lambda x: x[1], reverse=True)
            all_results[label] = results[:k]

        max_frame = max([frame for results in all_results.values() for frame, _, _ in results])
        new_data = [{'frame': i} for i in range(max_frame + 1)]

        for label, results in all_results.items():
            for frame, score, box in results:
                if 'scores' not in new_data[frame]:
                    new_data[frame]['scores'] = []
                    new_data[frame]['boxes'] = []
                    new_data[frame]['labels'] = []
                new_data[frame]['scores'].append(score)
                new_data[frame]['boxes'].append(box)
                new_data[frame]['labels'].append(label)
    else:
        print("Invalid run_type: must be image or lang ")
        return data['result']

    return new_data

def TopK_Visualization(data, input_video:str, run_type:str, output_folder='results/output_images'):
    if run_type == 'image':
        # Henry cannot find the bug(lmao).
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(input_video)

        for item in data['result']:
            if 'boxes' in item and len(item['boxes']) > 0:
                frame_num = item['frame']
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Unable to read frame {frame_num}")
                    continue

                print(f"Processing frame {frame_num}")

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                fig, ax = plt.subplots(1)

                ax.imshow(image)

                plt.axis('off')

                for box, score in zip(item['boxes'], item['scores']):
                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                             edgecolor='r', facecolor='none')

                    ax.add_patch(rect)
                    plt.text(box[0], box[1], str(score), color='r')

                plt.savefig(os.path.join(output_folder, f"frame_{frame_num}.png"), bbox_inches='tight', pad_inches=0)
                print(f"Saved image for frame {frame_num}")

                plt.close(fig)

        cap.release()
    elif run_type == 'lang':
        # still need testing
        color_dict = [(255 - round(255 / i), round(255 / i)) for i in range(len(data['query']) + 1, 1, -1)]
        class_string = ", ".join([f"{index + 1}->{c}" for index, c in enumerate(data['query'])])
        format_string = f"Classes: [{class_string}]"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(input_video)

        for item in data['result']:
            if 'boxes' in item and len(item['boxes']) > 0:
                frame_num = item['frame']
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Unable to read frame {frame_num}")
                    continue

                print(f"Processing frame {frame_num}")

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                cv2.putText(image, format_string, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                fig, ax = plt.subplots(1)

                ax.imshow(image)

                plt.axis('off')

                for box, score, label in zip(item['boxes'], item['scores'], item['labels']):
                    # BUG when using color_dict!!! 
                    color = 'w' # color_dict[label]

                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                             edgecolor=color, facecolor='none')

                    ax.add_patch(rect)
                    plt.text(box[0], box[1], f'{score}, label: {data["query"][label]}', color=color)

                plt.savefig(os.path.join(output_folder, f"frame_{frame_num}.png"), bbox_inches='tight', pad_inches=0)
                print(f"Saved image for frame {frame_num}")

                plt.close(fig)

        cap.release()
    else:
        print("Invalid run_type: must be image or lang ")

if __name__ == "__main__":
    input_file = 'results/IMG_1752.mp4_img.json'  # name of the file to be processed

    with open(input_file, 'r') as f:
        data = json.load(f)

    k = 50  # set a k
    # choose the type of query（img/lang）
    type = 'image'  # select the type (image/lang)
    new_result = top_k_results(data, k, type)

    data['result'] = new_result

    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_topk.json"

    with open(output_file, 'w') as f:
        json.dump(data, f)

    input_video = 'results/IMG_1752.mp4'  # name of the video to be processed
    TopK_Visualization(data, input_video, type)

    print("done")
