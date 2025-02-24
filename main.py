from body_detection import detect_body
from virtual_try_on import virtual_try_on

def main():
    print("1. Body Detection")
    print("2. Virtual Try-On")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        detect_body()
    elif choice == "2":
        # Use raw string or forward slashes
        clothing_path = r"C:\Users\Rohit\.vscode\AI_dressing_room\Black-T-Shirt-PNG-HD.png"
        virtual_try_on(clothing_path)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()