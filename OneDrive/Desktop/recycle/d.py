import tkinter as tk
from tkinter import messagebox

class Item:
    """
    Class representing a recyclable item.
    """
    def __init__(self, item_type, reward, shape=None, weight=None):
        self.item_type = item_type
        self.reward = reward
        self.shape = shape
        self.weight = weight

class RecyclingSystem:
    """
    Class managing the recycling system.
    """
    def __init__(self):
        self.items = []
        self.total_reward = 0.0
        self.item_rewards = {
            'A': 0.10,
            'B': 0.05,
            'C': 0.15
        }

    def add_item(self, item_type, shape=None, weight=None):
        """
        Add an item to the system and update the total reward.
        
        Parameters:
        item_type (str): The type of the item (A, B, C).
        shape (str): The shape of the item (optional).
        weight (float): The weight of the item (optional).
        
        Returns:
        str: Result message indicating success or failure.
        """
        if item_type in self.item_rewards:
            reward = self.item_rewards[item_type]
            item = Item(item_type, reward, shape, weight)
            self.items.append(item)
            self.total_reward += reward
            return f"Added item of type {item_type} with reward {reward} INR"
        else:
            return "Invalid item type. Please enter A, B, or C."

    def view_total_reward(self):
        """
        View the total reward accumulated in the system.
        
        Returns:
        float: Total reward.
        """
        return self.total_reward

    def reset_system(self):
        """
        Reset the system by clearing all items and resetting the total reward.
        
        Returns:
        str: Result message indicating the system has been reset.
        """
        self.items = []
        self.total_reward = 0.0
        return "System has been reset."

class RecyclingSystemGUI:
    """
    GUI class for the recycling system using Tkinter.
    """
    def __init__(self, root, system):
        self.root = root
        self.system = system
        self.root.title("Recycling System")

        self.create_widgets()
        
    def create_widgets(self):
        """
        Create and arrange the widgets in the GUI.
        """
        self.label_item_type = tk.Label(self.root, text="Enter item type (A, B, C):")
        self.label_item_type.pack(pady=5)

        self.entry_item_type = tk.Entry(self.root)
        self.entry_item_type.pack(pady=5)

        self.label_shape = tk.Label(self.root, text="Enter item shape:")
        self.label_shape.pack(pady=5)

        self.entry_shape = tk.Entry(self.root)
        self.entry_shape.pack(pady=5)

        self.label_weight = tk.Label(self.root, text="Enter item weight:")
        self.label_weight.pack(pady=5)

        self.entry_weight = tk.Entry(self.root)
        self.entry_weight.pack(pady=5)

        self.add_button = tk.Button(self.root, text="Add Item", command=self.add_item)
        self.add_button.pack(pady=5)

        self.view_button = tk.Button(self.root, text="View Total Reward", command=self.view_total_reward)
        self.view_button.pack(pady=5)

        self.reset_button = tk.Button(self.root, text="Reset System", command=self.reset_system)
        self.reset_button.pack(pady=5)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=5)

        self.result_label = tk.Label(self.root, text="", fg="blue")
        self.result_label.pack(pady=5)

    def add_item(self):
        """
        Handle the action of adding an item to the system.
        """
        item_type = self.entry_item_type.get().upper().strip()
        shape = self.entry_shape.get().strip()
        weight = float(self.entry_weight.get().strip())
        result = self.system.add_item(item_type, shape, weight)
        self.result_label.config(text=result)
        self.entry_item_type.delete(0, tk.END)
        self.entry_shape.delete(0, tk.END)
        self.entry_weight.delete(0, tk.END)

    def view_total_reward(self):
        """
        Display the total reward in a message box.
        """
        total_reward = self.system.view_total_reward()
        messagebox.showinfo("Total Reward", f"Total reward accumulated: {total_reward:.2f} INR")

    def reset_system(self):
        """
        Handle the action of resetting the system.
        """
        result = self.system.reset_system()
        self.result_label.config(text=result)

def main():
    """
    Main function to run the Tkinter GUI application.
    """
    root = tk.Tk()
    system = RecyclingSystem()
    app = RecyclingSystemGUI(root, system)
    root.mainloop()

if __name__ == "__main__":
    main()
